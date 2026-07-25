# esp32-edge-playground

A modular collection of ESP32-based AI, vision, IoT, and edge-inference projects.  
This repository is designed as a long-term workspace for experimenting with embedded ML, cloud inference, TinyML, camera pipelines, and agentic automation using ESP32-CAM and related microcontrollers.

---

## Overview

This repository contains multiple ESP32 projects, each structured as a standalone module with:

- Clear architecture diagrams  
- Reproducible workflows  
- Modular code  
- Cloud-ready integration points  
- Optional edge inference (Pico, Pi Zero, TinyML)  
- Vision pipelines (ESP32-CAM)  
- ML model deployment examples (FastAPI, Azure, TFLite)

The goal is to make it easy to prototype, extend, and scale ESP32-based AI systems.

---



## Project Index



### 1. ESP32-CAM → FastAPI MNIST Inference (Laptop / Cloud)

A complete pipeline where the ESP32-CAM captures an image, preprocesses it (grayscale → threshold → resize to 28×28), and sends it to a FastAPI inference server running on a laptop or Cloud say Azure or AWS.

### Why This Project?

ESP32 is a microcontroller, not a compute platform — and that constraint is the point.
This project explores how far you can push embedded hardware before the architecture
has to change, applying lean design principles to a real end-to-end AI pipeline.

The answer came quickly: standard Keras inference is not viable on the ESP32. The full
TF runtime and model weights (even a simple MNIST model sits in the hundreds of KB)
together exceed the available SRAM once the camera and WiFi stack are loaded. The
practical architecture is the ESP32 as a streaming camera device with inference
offloaded to an external server — local or cloud.

Firmware design turned out to be as demanding as the ML pipeline. The ESP32's 520 KB
of SRAM is shared between the camera frame buffers, the WiFi stack, and the
application — and the order in which you initialise these matters. Connecting to WiFi
before starting the camera, calling WiFi.setSleep(false) to prevent modem sleep stalls,
and detecting PSRAM to offload frame buffers all proved necessary to avoid random
crashes and connection freezes. Separating the MJPEG stream onto Core 0 and the
capture-and-inference loop onto Core 1 using FreeRTOS tasks was the key architectural
move that made both run reliably in parallel — without it, the blocking stream loop
starved the capture endpoint entirely.

What the project made clear is how many unglamorous challenges exist beyond the model
itself: optimal camera settings, subject positioning, lighting conditions, and text
size all directly affect prediction accuracy. To study these systematically, I went old-school and built an adjustable camera rig from popsicle sticks to vary distance, angle, and lighting in a controlled way.

**What's next:**  I learnt that TFLite Micro is specifically designed to run quantised models on microcontrollers and may make on-device inference viable after all. The plan is to test a reverse architecture — ESP32 as the inference server, laptop webcam as the streaming client — to directly compare accuracy, latency, and resource headroom against the current setup. Or even a second ESP32 CAM as client.

### Architecture — How It Evolved



#### What I planned

*ESP32 handles everything up to the model input*

```

+------------------+        Wi‑Fi (HTTP POST)        +------------------------+
|   ESP32‑CAM      |  -----------------------------> |  Laptop (Python API)   |
|                  |                                 |                        |
| Capture image    |                                 | Receive 28×28 array    |
| Grayscale        |                                 | Normalize              |
| Threshold        |                                 | Run MNIST model        |
| Resize to 28×28  |                                 | Display prediction     |
| Send array       |                                 |                        |
+------------------+                                 +------------------------+

```



#### What it actually ended-up

*After learning what the hardware could and couldn't do*

```
+--------------------------------------------------+
|                  ESP32-CAM                       |
|                                                  |
|  Init sequence (order matters to avoid crashes): |
|  WiFi -> setSleep(false) -> Camera (PSRAM-aware, |
|  init at UXGA, drop to QVGA) -> Servers          |
|                                                  |
|  +-- Core 0 (FreeRTOS task) -----------------+  |
|  |  MJPEG stream                             |  |
|  |  QVGA 320x240, CAMERA_GRAB_LATEST         |  |
|  |  WiFiServer :81, setNoDelay(true)         |  |
|  +-------------------------------------------+  |
|                          |                       |
|                          | raw MJPEG frames      |
|                          v                       |
|                     Browser :81                  |
|                   (live preview for              |
|                    subject positioning)          |
|                                                  |
|  +-- Core 1 (main loop) -------------------+    |
|  |  /capture endpoint (WebServer :80)      |    |
|  |  Physical button trigger (pin 14)       |    |
|  |                                         |    |
|  |  On capture:                            |    |
|  |  (1) Switch sensor to 96x96             |    |
|  |  (2) Discard stale frame                |    |
|  |  (3) Grab clean 96x96 JPEG             |    |
|  |  (4) HTTP POST raw JPEG to :8000        |    |
|  |  (5) Restore QVGA for stream            |    |
|  +-----------------------------------------+    |
+--------------------------------------------------+
                       |
          HTTP POST — raw JPEG bytes
          (not preprocessed, not an array)
                       |
                       v
+--------------------------------------------------+
|           Laptop — FastAPI :8000                 |
|                                                  |
|  /cleanup (async POST)                           |
|  +-- Preprocessing pipeline -----------------+  |
|  |  1.  Decode JPEG bytes -> grayscale        |  |
|  |      [ save: original.png ]                |  |
|  |  2.  Binary threshold at 127               |  |
|  |      dark ink -> 0,  background -> 255     |  |
|  |  3.  Morphological closing (5x5 kernel)    |  |
|  |      fills JPEG block artifacts in strokes |  |
|  |  4.  Find external contours                |  |
|  |  5.  Filter contours:                      |  |
|  |      - discard if touching image border    |  |
|  |        (removes shadow stripes/paper edges)|  |
|  |      - discard if too small (noise)        |  |
|  |      - discard if too large (> 85% frame)  |  |
|  |  6.  Select largest bounding rect          |  |
|  |  7.  Proportional padding (copyMakeBorder) |  |
|  |      [ save: cropped.png ]                 |  |
|  |  8.  Gaussian blur                         |  |
|  |  9.  Resize to 28x28                       |  |
|  |  10. Invert: white digit / black background|  |
|  |      (MNIST convention)                    |  |
|  |      [ save: debug.png ]                   |  |
|  +--------------------------------------------+  |
|                       |                          |
|                       v                          |
|  +-- MNIST model (.h5) --------------------+    |
|  |  flatten() -> /255.0 -> predict()       |    |
|  |  argmax -> {"digit": N}                 |    |
|  +-----------------------------------------+    |
|                       |                          |
|                       v                          |
|              /ui (HTML response)                 |
+--------------------------------------------------+
                       |
               serves browser UI
                       |
                       v
+--------------------------------------------------+
|            Browser — :8000/ui                    |
|                                                  |
|  <img src="ESP32:81">  <-- live MJPEG feed       |
|                            (position subject     |
|                             before capture)      |
|                                                  |
|  [Capture] button                                |
|    -> GET  ESP32:80/capture                      |
|    -> ESP32 POSTs raw JPEG to :8000/cleanup      |
|    -> receives {"digit": N}                      |
|    -> displays prediction                        |
|    -> resumes stream after 400ms                 |
+--------------------------------------------------+
```

**What's next — Reverse Architecture (planned)**

```

+--------------------------------------------------+
|           ESP32-CAM (inference server)           |
|  TFLite Micro model running on-device            |
|  Receives JPEG from client -> runs inference     |
|  Returns {"digit": N}                            |
+--------------------------------------------------+
                       ^
          HTTP POST — raw JPEG bytes
                       |
+--------------------------------------------------+
|         Laptop — webcam client                   |
|  Captures frames via OpenCV                      |
|  POSTs to ESP32 inference endpoint               |
|  Displays prediction                             |
+--------------------------------------------------+

```

**[esp32cam-mnist-fastapi](./esp32cam-mnist-fastapi):**

- ESP32-CAM capture + preprocessing  
- JSON payload transmission  
- FastAPI inference server  
- MNIST model (Keras/TFLite)  
- Cloud deployment notes (Azure App Service)

---



### 2. ESP32-CAM → Raspberry Pi Zero W (Edge Inference)

A fully offline edge-inference setup where the Pi Zero W runs a full MNIST CNN and the ESP32-CAM acts as the capture device.

**Includes:**

- Pi Zero W inference server  
- ESP32 → Pi communication  
- Vision preprocessing pipeline

---



### 3. ESP32-CAM → Raspberry Pi Pico (TinyML)

A TinyML-focused project using TFLite Micro on the Pico to run extremely small MNIST-style models.

**Includes:**

- TinyML model training  
- Quantization pipeline  
- ESP32 → Pico communication (UART/SPI/I²C/Wi-Fi for Pico W)

---



### 4. ESP32 Vision Utilities

Reusable modules for:

- Grayscale conversion  
- Thresholding  
- ROI cropping  
- Nearest-neighbor resizing  
- JPEG decoding  
- 28×28 preprocessing utilities

---



### 5. Cloud Integration Examples

Templates for connecting ESP32 devices to cloud inference endpoints.

**Includes:**

- FastAPI deployment  
- Azure App Service setup  
- HTTPS communication from ESP32  
- API key authentication patterns

---



## Repository Structure

```
esp32-edge-playground/
│
├── esp32cam-mnist-fastapi/
│   ├── esp32/                # ESP32-CAM firmware
│   ├── server/               # FastAPI inference server
│   ├── model/                # MNIST model files
│   └── docs/                 # Architecture diagrams
│
├── esp32cam-pizero-mnist/
│   ├── esp32/
│   ├── pizero/
│   └── docs/
│
├── esp32cam-pico-tinyml/
│   ├── esp32/
│   ├── pico/
│   └── tinyml/
│
├── vision-utils/
│   ├── grayscale/
│   ├── threshold/
│   ├── resize/
│   └── roi/
│
└── cloud-examples/
├── fastapi/
├── azure-appservice/
└── esp32-https/
```

---



## Getting Started



### Prerequisites

- ESP32-CAM module  
- Python 3.10+  
- FastAPI + Uvicorn  
- TensorFlow or TFLite  
- Optional: Raspberry Pi Zero W / Pico / Pico W  
- Optional: Azure subscription

---



### Setup (FastAPI MNIST Server)

1. Train or download a MNIST model (`model.h5`)
2. Place it in the `model/` directory
3. Run the FastAPI server:

```
uvicorn mnist_server:app --host 0.0.0.0 --port 8000

```

1. Configure ESP32-CAM to POST to:

```
http://<your-ip>:8000/predict
```

Or, after cloud deployment:

```
https://<your-azure-app>.azurewebsites.net/predict
```

---



## ESP32-CAM → Cloud Communication

The ESP32 sends a JSON payload:

```json
{
  "pixels": [784 grayscale values]
}
```

The server returns:

```json
{
  "digit": 7
}
```

This pattern is reused across all projects.

Roadmap
Add support for YOLO-lite models

Add ESP32-S3 support (better ML acceleration)

Add Azure Functions serverless inference

Add Pico W Wi-Fi TinyML examples

Add agentic workflows for automated data collection

Add multi-digit OCR pipeline

Add dataset generation tools using ESP32-CAM

License
MIT License (or your preferred license).

Contributions
Pull requests are welcome.
Each project is designed to be modular — feel free to add new ESP32-based AI or IoT modules.