"""
FHDSS SPRINT 25 — Phase 1 Test: Inference Preprocessing Contract Audit
Validates that eval-time data augmentation pipeline and production inference
preprocessing produce identical normalized tensors (Delta < 1e-6).
"""
import pytest
import numpy as np
import torch
from pathlib import Path
import cv2

from ml.preprocessing.contract import preprocess_for_inference, V4ForensicAugmentation

def test_inference_contract_random_shapes():
    aug_eval = V4ForensicAugmentation(is_training=False)
    
    test_shapes = [
        (224, 224, 3),
        (100, 300, 3),
        (600, 200, 3),
        (500, 500, 3),
        (150, 400, 3),
    ]
    
    for sh in test_shapes:
        np.random.seed(42)
        img_bgr = np.random.randint(0, 256, sh, dtype=np.uint8)
        t_aug = aug_eval(img_bgr, is_bgr=True).unsqueeze(0)
        t_inf = preprocess_for_inference(img_bgr, is_bgr=True)
        
        diff = float(torch.max(torch.abs(t_aug - t_inf)).item())
        assert diff < 1e-6, f"Shape {sh} BGR preprocessing mismatch: {diff}"
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t_aug_rgb = aug_eval(img_rgb, is_bgr=False).unsqueeze(0)
        t_inf_rgb = preprocess_for_inference(img_rgb, is_bgr=False)
        
        diff_rgb = float(torch.max(torch.abs(t_aug_rgb - t_inf_rgb)).item())
        assert diff_rgb < 1e-6, f"Shape {sh} RGB preprocessing mismatch: {diff_rgb}"

def test_inference_contract_real_sample():
    aug_eval = V4ForensicAugmentation(is_training=False)
    cedar_sample = Path("datasets/DS_Downloaded/CEDAR/CEDAR1/signatures/full_org/original_1_1.png")
    if cedar_sample.exists():
        data = cedar_sample.read_bytes()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None
        
        t_aug = aug_eval(img, is_bgr=True).unsqueeze(0)
        t_inf = preprocess_for_inference(img, is_bgr=True)
        
        diff = float(torch.max(torch.abs(t_aug - t_inf)).item())
        assert diff < 1e-6, f"Real image contract mismatch: {diff}"
