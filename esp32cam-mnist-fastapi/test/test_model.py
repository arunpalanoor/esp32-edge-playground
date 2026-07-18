import requests
import numpy as np

# Fake test image
img = np.random.randint(0, 255, (28, 28)).flatten().tolist()

r = requests.post("http://localhost:8000/predict", json={"pixels": img})
print(r.json())
