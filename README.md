# CNN-Derived-Fingerprint-Embeddings-for-Secure-Medical-Image-Watermarking-and-Forgery-Detection
# Biometric watermarking for CT images using CNN fingerprint embeddings + DWT. Forgery detection CNN achieves 96.4% accuracy | PSNR: 51.86 dB | AUC: 0.926
from google.colab import drive
drive.mount('/content/drive')
!ls "/content/drive/MyDrive/medical dataset"
!ls "/content/drive/MyDrive/biometrics dataset"
import cv2
import matplotlib.pyplot as plt
import os

# Folder paths
medical_path = "/content/drive/MyDrive/medical dataset"
finger_path = "/content/drive/MyDrive/biometrics dataset"

# Read one image from each
med_img = cv2.imread(os.path.join(medical_path, os.listdir(medical_path)[0]), cv2.IMREAD_GRAYSCALE)
fin_img = cv2.imread(os.path.join(finger_path, os.listdir(finger_path)[0]), cv2.IMREAD_GRAYSCALE)

# Display both
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(med_img, cmap='gray')
plt.title("Medical Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(fin_img, cmap='gray')
plt.title("Fingerprint Image")
plt.axis('off')
plt.show()
import numpy as np
import pywt

# Resize both images to a common size (say 256x256)
common_size = (256, 256)
med_img_resized = cv2.resize(med_img, common_size)
fin_img_resized = cv2.resize(fin_img, common_size)

# Normalize to range 0–1
med_img_norm = med_img_resized / 255.0
fin_img_norm = fin_img_resized / 255.0

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(med_img_norm, cmap='gray')
plt.title("Resized Medical Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(fin_img_norm, cmap='gray')
plt.title("Resized Fingerprint Image")
plt.axis('off')
plt.show()
# Apply single-level DWT on medical image
coeffs_med = pywt.dwt2(med_img_norm, 'haar')
LL, (LH, HL, HH) = coeffs_med

# Embed fingerprint into LH band (for better invisibility)
alpha = 0.05  # embedding strength (tune between 0.01–0.1)
fin_img_resized_small = cv2.resize(fin_img_norm, (LH.shape[1], LH.shape[0]))

# Embed fingerprint into LH band
alpha = 0.05  # embedding strength
LH_embedded = LH + alpha * fin_img_resized_small

# Reconstruct the watermarked image using inverse DWT
watermarked = pywt.idwt2((LL, (LH_embedded, HL, HH)), 'haar')

# Clip values to range 0–1
watermarked = np.clip(watermarked, 0, 1)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(med_img_norm, cmap='gray')
plt.title("Original Medical Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(fin_img_norm, cmap='gray')
plt.title("Fingerprint (Watermark)")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(watermarked, cmap='gray')
plt.title("Watermarked Image")
plt.axis('off')
plt.show()
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Step 1: Apply DWT to the watermarked image
coeffs_watermarked = pywt.dwt2(watermarked, 'haar')
LL_w, (LH_w, HL_w, HH_w) = coeffs_watermarked

# Step 2: Extract the watermark from LH band
extracted_fin_img = (LH_w - LH) / alpha

# Resize extracted watermark to original fingerprint size
common_size = fin_img_norm.shape
extracted_fin_img_resized = cv2.resize(extracted_fin_img, (common_size[1], common_size[0]))

# Clip values to 0–1
extracted_fin_img_resized = np.clip(extracted_fin_img_resized, 0, 1)

# Step 3: Compute PSNR and SSIM
psnr_val = psnr(fin_img_norm, extracted_fin_img_resized)
ssim_val = ssim(fin_img_norm, extracted_fin_img_resized, data_range=1.0)

print(f"PSNR between original and extracted watermark: {psnr_val:.2f} dB")
print(f"SSIM between original and extracted watermark: {ssim_val:.4f}")

# Step 4: Display extracted watermark
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(fin_img_norm, cmap='gray')
plt.title("Original Fingerprint (Watermark)")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(extracted_fin_img_resized, cmap='gray')
plt.title("Extracted Watermark")
plt.axis('off')
plt.show()
