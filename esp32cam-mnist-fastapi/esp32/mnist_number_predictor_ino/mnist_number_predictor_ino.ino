#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// ====== CAMERA CONFIG ======
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

// ====== LED + BUTTON ======
#define LED_PIN 4        // Built‑in flash LED (safe)
#define BUTTON_PIN 14    // Safe button pin

// ====== WIFI CONFIG ======
const char *ssid = "**************";
const char *password = "*************";

// ====== API CONFIG ======
String serverUrl = "http://:8000/predict";

// ====== Resize target ======
const int TARGET_W = 28;
const int TARGET_H = 28;

// ====== Helper: Convert RGB565 to grayscale ======
uint8_t rgb565_to_gray(uint16_t pixel) {
    uint8_t r = (pixel >> 11) & 0x1F;
    uint8_t g = (pixel >> 5) & 0x3F;
    uint8_t b = pixel & 0x1F;

    r <<= 3;
    g <<= 2;
    b <<= 3;

    return (uint8_t)(0.299*r + 0.587*g + 0.114*b);
}

// ====== LED Blink Helper ======
void blinkLED(int count) {
    for (int i = 0; i < count; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(200);
        digitalWrite(LED_PIN, LOW);
        delay(200);
    }
}

// ====== Resize using nearest-neighbor ======
void resizeNN(uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    float x_ratio = sw / (float)dw;
    float y_ratio = sh / (float)dh;

    for (int j = 0; j < dh; j++) {
        for (int i = 0; i < dw; i++) {
            int px = (int)(i * x_ratio);
            int py = (int)(j * y_ratio);
            dst[j * dw + i] = src[py * sw + px];
        }
    }
}

// ====== Capture + Process + Send ======
void capture_and_send() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return;
    }

    int srcW = fb->width;
    int srcH = fb->height;

    uint8_t* gray = (uint8_t*)malloc(srcW * srcH);
    uint16_t* pixels = (uint16_t*)fb->buf;

    for (int i = 0; i < srcW * srcH; i++) {
        gray[i] = rgb565_to_gray(pixels[i]);
    }

    uint8_t resized[TARGET_W * TARGET_H];
    resizeNN(gray, srcW, srcH, resized, TARGET_W, TARGET_H);

    free(gray);
    esp_camera_fb_return(fb);

    // Invert polarity (MNIST expects white digit on black)
    for (int i = 0; i < TARGET_W * TARGET_H; i++) {
        resized[i] = 255 - resized[i];
    }

    // Build JSON
    String json = "{\"pixels\":[";
    for (int i = 0; i < TARGET_W * TARGET_H; i++) {
        json += String(resized[i]);
        if (i < TARGET_W * TARGET_H - 1) json += ",";
    }
    json += "]}";

    // Send to FastAPI
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(serverUrl);
        http.addHeader("Content-Type", "application/json");

        int code = http.POST(json);
        if (code > 0) {
            Serial.println("Prediction: " + http.getString());
        } else {
            Serial.println("POST failed");
        }
        http.end();
    }
}

void setup() {
    Serial.begin(115200);
    delay(2000);

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    pinMode(BUTTON_PIN, INPUT_PULLUP);

    // ===== WiFi Connect =====
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nConnected!");

    // ===== Blink twice for WiFi connected =====
    blinkLED(2);

    // ===== Camera init =====
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565;

    config.frame_size = FRAMESIZE_QQVGA;
    config.fb_count = 1;

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera init failed!");
        return;
    }

    Serial.println("Ready. Press button to capture.");
}

void loop() {
    if (digitalRead(BUTTON_PIN) == LOW) {
        delay(50); // debounce
        if (digitalRead(BUTTON_PIN) == LOW) {

            Serial.println("Button pressed, capturing...");
            capture_and_send();

            // ===== Blink 3 times for capture complete =====
            blinkLED(3);

            delay(500); // prevent double-trigger
        }
    }
}
