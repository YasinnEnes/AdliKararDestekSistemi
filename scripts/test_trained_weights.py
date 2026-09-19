import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
import torch
from pathlib import Path
import torchvision.transforms as T

from ml.models.siamese import ForensicConvNeXtEncoder, compute_embedding_distance
from scripts.debug_similarity import create_demo_img

weights_path = Path("artifacts/weights/convnext_tiny_forensic_best.pt")
encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)

if weights_path.exists():
    cp = torch.load(weights_path, map_location="cpu")
    if isinstance(cp, dict) and "model_state_dict" in cp:
        encoder.load_state_dict(cp["model_state_dict"])
    elif isinstance(cp, dict) and "encoder_state_dict" in cp:
        encoder.load_state_dict(cp["encoder_state_dict"])
    else:
        encoder.load_state_dict(cp, strict=False)
    print("Loaded weights successfully!")

encoder.eval()

img_q = create_demo_img(14.5, False)
img_r1 = create_demo_img(14.0, False)
img_r2 = create_demo_img(15.0, False)

# Different image (circles and blocks)
img_diff = np.ones((400, 600, 3), dtype=np.uint8) * 245
cv2.circle(img_diff, (300, 200), 80, (26, 26, 46), 4)
cv2.rectangle(img_diff, (100, 100), (200, 300), (26, 26, 46), 4)

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def preprocess_tensor(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 and img.shape[2] == 3 else img
    img_res = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img_res).permute(2, 0, 1).float() / 255.0
    t = normalize(t).unsqueeze(0)
    return t

with torch.inference_mode():
    eq = encoder(preprocess_tensor(img_q))
    er1 = encoder(preprocess_tensor(img_r1))
    er2 = encoder(preprocess_tensor(img_r2))
    ediff = encoder(preprocess_tensor(img_diff))

    cos1, ang1, norm1 = compute_embedding_distance(eq, er1)
    cos2, ang2, norm2 = compute_embedding_distance(eq, er2)
    cos_diff, ang_diff, norm_diff = compute_embedding_distance(eq, ediff)

print(f"Properly normalized Genuine Q vs R1: cos={cos1}, norm_sim={norm1}")
print(f"Properly normalized Genuine Q vs R2: cos={cos2}, norm_sim={norm2}")
print(f"Properly normalized Different Q vs Diff: cos={cos_diff}, norm_sim={norm_diff}")
