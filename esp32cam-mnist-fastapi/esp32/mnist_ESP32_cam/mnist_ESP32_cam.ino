#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>

// ====== CAMERA PINS (AI Thinker ESP32-CAM) ======
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ====== HARDWARE ======
#define LED_PIN    4
#define BUTTON_PIN 14

// ====== Clock frequency (MHz) ======
// OV2640 is rated to 20 MHz. Try 24 if you want to push for higher stream FPS.
// Do not exceed 24 — above that the sensor becomes unstable.
#define XCLK_FREQ_MHZ 24

// ====== WiFi ======
const char* ssid     = "your_SSID";
const char* password = "your_password";

// ====== FastAPI endpoint ======
const char* serverUrl = "http://192.168.0.158:8000/cleanup";

// ====== Servers ======
// Port 80  — /capture  (WebServer, non-blocking poll on Core 1)
// Port 81  — MJPEG stream (raw WiFiServer, FreeRTOS task on Core 0)
WebServer  captureServer(80);
WiFiServer streamServer(81);

// ====== Stream task state ======
WiFiClient   activeStreamClient;
TaskHandle_t streamTaskHandle = NULL;

// ======================================================
//  LED helper
// ======================================================
void blinkLED(int count) {
    for (int i = 0; i < count; i++) {
        digitalWrite(LED_PIN, HIGH); delay(150);
        digitalWrite(LED_PIN, LOW);  delay(150);
    }
}

// ======================================================
//  MJPEG stream task — pinned to Core 0
//  Reads from the global `activeStreamClient`.
//  Core 1 (loop) does not touch this client while
//  streamTaskHandle != NULL.
// ======================================================
void streamTask(void* pvParameters) {
    WiFiClient& client = activeStreamClient;
    char partHeader[128];

    // Wait up to 3 s for the browser's HTTP GET to arrive
    unsigned long deadline = millis() + 3000;
    while (client.connected() && !client.available()) {
        if (millis() > deadline) {
            client.stop();
            streamTaskHandle = NULL;
            vTaskDelete(NULL);
            return;
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    // Drain the request headers (not needed)
    while (client.available()) client.read();

    // HTTP + MJPEG response
    client.print(
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: keep-alive\r\n"
        "\r\n"
    );

    while (client.connected()) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        size_t hlen = snprintf(partHeader, sizeof(partHeader),
            "--frame\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: %u\r\n"
            "\r\n",
            (unsigned)fb->len
        );

        client.write((const uint8_t*)partHeader, hlen);
        client.write(fb->buf, fb->len);
        client.write((const uint8_t*)"\r\n", 2);

        esp_camera_fb_return(fb);
        vTaskDelay(pdMS_TO_TICKS(1));  // yield; no artificial FPS cap
    }

    client.stop();
    streamTaskHandle = NULL;
    vTaskDelete(NULL);
}

// ======================================================
//  Capture at 96×96 and POST raw JPEG to FastAPI
//
//  Why switch frame size?
//  The stream runs at QVGA (320×240) for a comfortable
//  preview. For the actual capture we switch down to
//  96×96 so the digit fills the whole frame, giving the
//  model a tighter, cleaner crop before it preprocesses
//  to 28×28. The UI already disconnects the stream
//  (img.src="") before calling /capture, so the 200 ms
//  stabilisation delay is also enough time for the stream
//  task to detect the disconnect and exit cleanly.
// ======================================================
String capture_and_send() {
    sensor_t* s = esp_camera_sensor_get();

    // Switch sensor to 96×96
    s->set_framesize(s, FRAMESIZE_96X96);
    delay(200);  // stabilise + let stream task exit

    // Discard one frame — likely a QVGA leftover in the buffer
    camera_fb_t* stale = esp_camera_fb_get();
    if (stale) esp_camera_fb_return(stale);

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        s->set_framesize(s, FRAMESIZE_QVGA);  // restore before returning
        blinkLED(5);
        return "{\"digit\": -1}";
    }

    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/octet-stream");

    int    code     = http.POST(fb->buf, fb->len);
    String response = (code > 0) ? http.getString() : "{\"digit\": -1}";

    http.end();
    esp_camera_fb_return(fb);

    // Restore QVGA for stream preview
    s->set_framesize(s, FRAMESIZE_QVGA);
    delay(100);

    return response;
}

// ====== /capture HTTP handler ======
void handleCapture() {
    blinkLED(1);
    String result = capture_and_send();
    blinkLED(3);
    captureServer.send(200, "application/json", result);
}

// ======================================================
//  Setup
// ======================================================
void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    Serial.println();

    pinMode(LED_PIN,    OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);  // LOW = pressed (button connects to GND)

    // WiFi — disable modem sleep for lower stream latency
    WiFi.begin(ssid, password);
    WiFi.setSleep(false);
    Serial.print("Connecting");
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
    Serial.println("\nConnected!");
    Serial.printf("  Stream:  http://%s:81\n",      WiFi.localIP().toString().c_str());
    Serial.printf("  Capture: http://%s/capture\n", WiFi.localIP().toString().c_str());
    blinkLED(2);

    // Camera config
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;

    config.xclk_freq_hz = XCLK_FREQ_MHZ * 1000000;  // set in MHz above
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode    = CAMERA_GRAB_LATEST;  // always grab freshest frame

    if (psramFound()) {
        Serial.println("PSRAM found — high quality mode");
        config.frame_size   = FRAMESIZE_UXGA;  // init large; downscaled below
        config.jpeg_quality = 10;
        config.fb_count     = 2;               // double-buffer in PSRAM
        config.fb_location  = CAMERA_FB_IN_PSRAM;
    } else {
        Serial.println("No PSRAM — SVGA fallback");
        config.frame_size   = FRAMESIZE_SVGA;
        config.jpeg_quality = 12;
        config.fb_count     = 1;
        config.fb_location  = CAMERA_FB_IN_DRAM;
    }

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera init failed!");
        blinkLED(6);
        return;
    }

    // Sensor tuning for handwriting on white paper
    sensor_t* s = esp_camera_sensor_get();
    s->set_framesize(s,  FRAMESIZE_QVGA);  // 320×240 stream preview
    s->set_quality(s,    12);
    s->set_brightness(s,  1);   // slightly brighter for indoor light
    s->set_contrast(s,    2);   // high contrast → sharper digit edges
    s->set_saturation(s, -2);   // near-grayscale, matches MNIST input
    s->set_sharpness(s,   2);   // crispens pen strokes

    captureServer.on("/capture", HTTP_GET, handleCapture);
    captureServer.begin();

    streamServer.begin();
    streamServer.setNoDelay(true);  // disable Nagle → lower frame latency

    Serial.println("Ready.");
}

// ======================================================
//  Loop — Core 1
// ======================================================
void loop() {
    // ---- Physical button: trigger capture with debounce ----
    if (digitalRead(BUTTON_PIN) == LOW) {
        delay(50);                               // debounce pause
        if (digitalRead(BUTTON_PIN) == LOW) {    // still pressed → real press
            Serial.println("Button — capturing");
            blinkLED(1);
            String result = capture_and_send();
            Serial.println(result);
            blinkLED(3);
            while (digitalRead(BUTTON_PIN) == LOW) delay(10);  // wait for release
        }
    }

    // ---- Accept new stream client on port 81 ----
    if (streamServer.hasClient()) {
        if (streamTaskHandle != NULL) {
            // Already streaming — reject the new connection
            WiFiClient rejected = streamServer.accept();
            rejected.stop();
        } else {
            activeStreamClient = streamServer.accept();
            activeStreamClient.setNoDelay(true);
            xTaskCreatePinnedToCore(
                streamTask, "CamStream", 4096, NULL,
                5, &streamTaskHandle, 0  // Core 0
            );
        }
    }

    captureServer.handleClient();
    delay(1);  // yield to FreeRTOS scheduler
}
