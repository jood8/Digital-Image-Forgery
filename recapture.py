
import cv2
import numpy as np
import os
from pathlib import Path

def apply_recapture_effects(image):
    # محاكاة خطوط الشاشة 
    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    # معادلة رياضية لعمل تموجات خفيفة
    moire = np.sin(x * 0.2) + np.sin(y * 0.2) 
    moire = cv2.merge([moire, moire, moire])
    
    # دمج التموجات مع الصورة الأصلية
    simulated = cv2.addWeighted(image.astype(float), 0.9, moire * 15, 0.1, 0)
    
    # إضافة Sensor Noise 
    noise = np.random.normal(0, 3, (h, w, 3))
    simulated = simulated + noise
    
    return np.clip(simulated, 0, 255).astype(np.uint8)

input_root = r"natural_images"
output_root = r"simulated_recaptured"

if not os.path.exists(output_root):
    os.makedirs(output_root)

print("Starting Simulation... Please wait.")

for subdir in os.listdir(input_root):
    sub_path = os.path.join(input_root, subdir)
    if os.path.isdir(sub_path):
        out_sub_path = os.path.join(output_root, subdir)
        os.makedirs(out_sub_path, exist_ok=True)
        
        count = 0
        for img_name in os.listdir(sub_path):
            if count >= 500: break 
            
            img_path = os.path.join(sub_path, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                recaptured_img = apply_recapture_effects(img)
                cv2.imwrite(os.path.join(out_sub_path, img_name), recaptured_img)
                count += 1

print("Done! You now have a balanced dataset.")


import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

def show_lbp_comparison(original_path, simulated_path):
    orig = cv2.imread(original_path, 0)
    sim  = cv2.imread(simulated_path, 0)
    
    # حساب الـ LBP 
    lbp_orig = local_binary_pattern(orig, 8, 1, method='default')
    lbp_sim  = local_binary_pattern(sim, 8, 1, method='default')
    
    # عرض النتائج للمقارنة
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1); plt.imshow(lbp_orig, cmap='gray'); plt.title('LBP Original')
    plt.subplot(1, 2, 2); plt.imshow(lbp_sim, cmap='gray'); plt.title('LBP Simulated (Recaptured)')
    plt.show()

show_lbp_comparison('natural_images/airplane/airplane_0000.jpg',
 'simulated_recaptured/airplane/airplane_0000.jpg')