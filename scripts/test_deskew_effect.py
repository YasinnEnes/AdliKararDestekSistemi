import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from cv.preprocessing.pipeline import ForensicImagePreprocessor
from cv.features.extractors import ForensicFeatureExtractor
from scripts.debug_similarity import create_demo_img

preprocessor_with_deskew = ForensicImagePreprocessor(deskew_enabled=True)
preprocessor_no_deskew = ForensicImagePreprocessor(deskew_enabled=False)

img = create_demo_img(14.5, False)

art1 = preprocessor_with_deskew.process(img, deskew_enabled=True)
art2 = preprocessor_no_deskew.process(img, deskew_enabled=False)

print("Deskew angle estimated:", art1.metadata.get("deskew_angle_deg"))
print("With deskew skeleton nonzeros:", np.count_nonzero(art1.skeleton))
print("Without deskew skeleton nonzeros:", np.count_nonzero(art2.skeleton))

extractor = ForensicFeatureExtractor()
f1 = extractor.extract_all(art1.skeleton, art1.binarized)
f2 = extractor.extract_all(art2.skeleton, art2.binarized)

print("\n--- WITH DESKEW ---")
print("Terminals:", f1.structural.terminal_count)
print("Crossings:", f1.structural.crossing_count)
print("Loops:", f1.structural.loop_count)
print("Slant:", f1.structural.slant_angle_deg)
print("Avg stroke len:", f1.structural.avg_stroke_length_px)

print("\n--- WITHOUT DESKEW ---")
print("Terminals:", f2.structural.terminal_count)
print("Crossings:", f2.structural.crossing_count)
print("Loops:", f2.structural.loop_count)
print("Slant:", f2.structural.slant_angle_deg)
print("Avg stroke len:", f2.structural.avg_stroke_length_px)
