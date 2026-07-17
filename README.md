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
A complete pipeline where the ESP32-CAM captures an image, preprocesses it (grayscale → threshold → resize to 28×28), and sends it to a FastAPI inference server running on a laptop or Azure.

**End‑to‑end workflow**
```
+------------------+        Wi‑Fi (HTTP POST)        +------------------------+
|   ESP32‑CAM      |  -----------------------------> |   Laptop (Python API)  |
|                  |                                  |                        |
| Capture image    |                                  | Receive 28×28 array    |
| Grayscale        |                                  | Normalize              |
| Threshold        |                                  | Run MNIST model        |
| Resize to 28×28  |                                  | Return prediction      |
| Send array       |                                  |                        |
+------------------+                                  +------------------------+
```


**Includes:**
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


4. Configure ESP32-CAM to POST to:

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

