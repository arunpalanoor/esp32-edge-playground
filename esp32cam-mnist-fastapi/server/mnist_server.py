import json
import os
import io
import urllib.error
import urllib.request
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = FastAPI()

# Load MNIST model
model = tf.keras.models.load_model("../model/model.h5")

# ===== /predict schema (existing endpoint, unchanged) =====
class MNISTImage(BaseModel):
    pixels: list

# ===== Core prediction logic =====
def run_mnist_from_pixels(pixels: list) -> int:
    # FIX: removed plt.imshow() / plt.show() — calling plt.show() in a
    # FastAPI process blocks the server thread and crashes with no display.
    # If you want to debug visuals, use plt.savefig("debug.png") instead.
    arr = np.array(pixels, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 784)
    preds = model.predict(arr)
    return int(np.argmax(preds))

# ===== Original /predict (unchanged) =====
@app.post("/predict")
def predict_digit(data: MNISTImage):
    digit = run_mnist_from_pixels(data.pixels)
    return {"digit": digit}

# ===== JPEG → 28×28 preprocessing =====
# Techniques adapted from github.com/alankrantas/MNIST-Live-Detection-TFLite

# Contours touching within this many pixels of the image edge are discarded.
# Catches shadow stripes, paper edges, and other border artifacts without
# needing to know their shape. Keep small (5) so a digit near the edge survives.
BORDER = 5
MIN_DIM = 15    # contours smaller than this in either axis are noise — skip
MAX_FILL = 0.85 # contours filling more than 85% of the frame are wrong — skip

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

def save_debug_image(arr: np.ndarray, filename: str) -> None:
    """Save a grayscale numpy array as a clean PNG with no axes or whitespace."""
    plt.imshow(arr, cmap="gray")
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()

def preprocess_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    # Decode raw JPEG bytes → grayscale numpy array
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("L")
    img = np.array(img)
    img_h, img_w = img.shape

    # Save 1: raw image as received from ESP32
    save_debug_image(img, "original.png")

    # Step 1: Hard binary threshold — dark ink → 0, background → 255
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    save_debug_image(bw, "bw.png")

    # Step 2: Morphological closing on the inverted image (digit = white).
    # Fills small gaps and merges nearby stroke fragments into solid blobs
    # before contour detection. Especially useful for JPEG block artifacts
    # that can break a single stroke into disconnected pieces.
    inv = cv2.bitwise_not(bw)
    inv_closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, MORPH_KERNEL)

    # Step 3: Find all external contours in the closed image
    contours, _ = cv2.findContours(inv_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((28, 28), dtype=np.float32)

    # Step 4: Filter contours — discard border artifacts, noise, and oversized blobs.
    # Border filter: any contour whose bounding rect overlaps the image edge is
    # skipped. This cleanly removes shadow stripes (which start at x=0) and
    # paper edges without needing to know their shape or size in advance.
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (x < BORDER or y < BORDER
                or (x + w) > (img_w - BORDER)
                or (y + h) > (img_h - BORDER)):
            continue                          # touches image edge → shadow / paper edge
        if w < MIN_DIM or h < MIN_DIM:
            continue                          # too small → noise spot
        if w > img_w * MAX_FILL or h > img_h * MAX_FILL:
            continue                          # too large → likely not a digit
        candidates.append(c)

    if not candidates:
        return np.zeros((28, 28), dtype=np.float32)

    # Step 5: From surviving candidates pick the one with the largest bounding area
    best = max(candidates, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])
    x, y, w, h = cv2.boundingRect(best)

    # Step 6: Proportional square padding (adapted from reference repo).
    # r//5 adds a border proportional to the digit size rather than a fixed px value,
    # so small and large digits both get appropriate breathing room.
    # copyMakeBorder pads with white (255) to match the background in `bw`.
    r      = max(w, h)
    x_pad  = (((h - w) // 2) if h > w else 0) + r // 5   # centres if h>w, then adds border
    y_pad  = (((w - h) // 2) if w > h else 0) + r // 5   # centres if w>h, then adds border
    crop   = bw[y:y + h, x:x + w]

    # Save 2: tight crop around the digit (before padding/resize)
    save_debug_image(crop, "cropped.png")

    padded = cv2.copyMakeBorder(crop,
                                top=y_pad, bottom=y_pad,
                                left=x_pad, right=x_pad,
                                borderType=cv2.BORDER_CONSTANT,
                                value=255)   # white background

    # Step 7: Blur, resize, invert
    blurred  = cv2.GaussianBlur(padded, (3, 3), 0)
    resized  = cv2.resize(blurred, (28, 28), interpolation=cv2.INTER_AREA)
    inverted = 255 - resized   # white digit on black background (MNIST convention)

    # Save 3: final 28x28 image that will be fed to the model
    save_debug_image(inverted, "debug.png")

    return inverted.astype(np.float32)

# ===== UI / ESP32 config =====
ESP32_IP     = "http://192.168.0.243"       # ESP32 capture endpoint (port 80)
ESP32_STREAM = "http://192.168.0.243:81"    # MJPEG stream (port 81 in firmware)

# Set by /cleanup; used as fallback if ESP32 /capture body is missing digit.
_last_digit: int | None = None

# ===== /cleanup — receives raw JPEG from ESP32 =====
# FIX: must be `async def` so that `await request.body()` works.
# The original `def cleanup` + `request.body()` returned a coroutine object
# (not bytes), then fell back to `request._body` which is None until the
# body has been consumed — so jpeg_bytes was always None and every call crashed.
@app.post("/cleanup")
async def cleanup(request: Request):
    global _last_digit
    jpeg_bytes: bytes = await request.body()

    img28  = preprocess_jpeg(jpeg_bytes)
    pixels = img28.flatten().tolist()

    digit  = run_mnist_from_pixels(pixels)
    _last_digit = digit
    print({"digit": digit})
    return {"digit": digit}

# ===== /capture — browser calls this (same origin), server proxies ESP32 =====
# The UI cannot fetch ESP32 directly: the browser blocks cross-origin reads
# (localhost:8000 -> ESP32 LAN IP) even when the ESP32 request succeeds.
@app.get("/capture")
def capture_proxy():
    global _last_digit
    _last_digit = None

    try:
        with urllib.request.urlopen(f"{ESP32_IP}/capture", timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return {"digit": -1}

    digit = data.get("digit", -1)
    if digit == -1 and _last_digit is not None:
        digit = _last_digit
    return {"digit": digit}

# ===== UI =====

@app.get("/ui", response_class=HTMLResponse)
def ui():
    html = f"""
    <html>
    <head>
        <title>ESP32-CAM MNIST</title>
        <style>
            body         {{ background:#111; color:#eee; font-family:Arial; text-align:center; }}
            #container   {{ display:inline-flex; gap:30px; margin-top:20px; align-items:flex-start; }}
            #stream      {{ border:2px solid #444; }}
            #prediction  {{ font-size:72px; margin-top:20px; }}
            #label       {{ font-size:18px; color:#aaa; }}
            button       {{ padding:12px 28px; font-size:20px; cursor:pointer; margin-top:10px; }}
        </style>
    </head>
    <body>
        <h1>ESP32-CAM MNIST</h1>

        <div id="container">
            <!-- Stream is now on port 81 -->
            <img id="stream" src="{ESP32_STREAM}" width="320" height="240" />
            <div>
                <button onclick="capture()">Capture &amp; Predict</button>
                <div id="label">Prediction</div>
                <div id="prediction">-</div>
            </div>
        </div>

        <script>
            const streamEl = document.getElementById("stream");
            const predEl   = document.getElementById("prediction");

            function capture() {{
                // Pause the MJPEG stream while we POST the capture request.
                // Setting src="" closes the TCP connection to port 81 cleanly.
                streamEl.src = "";

                fetch("/capture")
                    .then(r => {{
                        if (!r.ok) throw new Error("capture failed");
                        return r.json();
                    }})
                    .then(data => {{
                        predEl.textContent = data.digit !== -1
                            ? data.digit
                            : "Error";

                        // Brief pause before re-opening the stream so the
                        // ESP32 has time to finish the /capture HTTP response
                        // before a new TCP connection hits port 81.
                        setTimeout(() => {{
                            streamEl.src = "{ESP32_STREAM}";
                        }}, 400);
                    }})
                    .catch(() => {{
                        predEl.textContent = "Error";
                        streamEl.src = "{ESP32_STREAM}";
                    }});
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)