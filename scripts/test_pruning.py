import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
import math

def prune_skeleton_spurs(skeleton: np.ndarray, max_spur_length: int = 4) -> np.ndarray:
    """
    Zhang-Suen iskeletindeki 1-piksel kenar gürültüsü kaynaklı mikro-kılçıkları (spurs) budar.
    """
    skel = (skeleton > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    
    for _ in range(max_spur_length):
        # 8-komşuluk toplamı
        neighbors = cv2.filter2D(skel, -1, kernel) * skel
        # Uç noktalar (neighbors == 1)
        endpoints = (neighbors == 1) & (skel == 1)
        if not np.any(endpoints):
            break
        # Uç noktaları sil
        skel[endpoints] = 0
    
    return (skel * 255).astype(np.uint8)

def calculate_forensic_slant(skeleton: np.ndarray) -> float:
    """
    Adli el yazısı eğim açısı (Slant):
    Dikey ve yarı-dikey iniş vuruşlarının (downstrokes) eğim açısını hesaplar.
    """
    skel_pts = np.argwhere(skeleton > 0)
    if len(skel_pts) < 15:
        return 0.0

    # Sobel gradyanı ile kontur doğrultuları
    sobelx = cv2.Sobel(skeleton.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(skeleton.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    
    # Gradyan açısı = çizgiye dik açı -> çizgi açısı = gradyan açısı + 90
    angles = np.arctan2(sobely, sobelx) # radyan
    angles_deg = np.degrees(angles)
    
    # Yalnızca iskelet piksellerindeki açıları filtrele
    mask = skeleton > 0
    skel_angles = angles_deg[mask]
    
    # Dikey iniş vuruşları (açısı 45 ile 135 veya -135 ile -45 derece arası)
    vertical_strokes = []
    for a in skel_angles:
        # Yataydan sapma açısını hesapla [-45, +45]
        # Dikey eksen (90 deg) referans
        diff = a - 90.0
        while diff > 90: diff -= 180
        while diff < -90: diff += 180
        if abs(diff) <= 45.0:
            vertical_strokes.append(-diff) # Sağa eğim pozitif (+)

    if len(vertical_strokes) > 5:
        return float(round(np.median(vertical_strokes), 2))
    return 0.0

print("Pruning and Slant functions compiled successfully.")
