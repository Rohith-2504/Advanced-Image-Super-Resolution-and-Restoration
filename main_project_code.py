# --- Cell 1: Setup Path ---
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/ESRGAN master')

# --- Cell 2: List Directory ---
!ls /content/drive/MyDrive/Colab Notebooks/ESRGAN master

# --- Cell 3: Architecture Upload ---
from google.colab import files
# files.upload()  # Manual upload step

# --- Cell 4: Import Architecture ---
import RRDBNet_arch as arch

# --- Cell 5: Core Processing Logic ---
import cv2
import numpy as np
import torch
import functools
import RRDBNet_arch as arch
from google.colab import files
import matplotlib.pyplot as plt

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def upload_and_process_image(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    
    uploaded = files.upload()
    if not uploaded:
        return

    file_name = list(uploaded.keys())[0]
    original_img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    img = original_img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    original_resized = cv2.resize(original_img, (output.shape[1], output.shape[0]))
    psnr_value = calculate_psnr(original_resized, output)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Enhanced")
    plt.show()

if __name__ == '__main__':
    model_path = '/content/drive/MyDrive/Colab Notebooks/ESRGAN master/models/RRDB_ESRGAN_x4.pth'
    upload_and_process_image(model_path)
