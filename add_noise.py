import numpy as np
import matplotlib.pyplot as plt
import cv2

def add_noise_to_image(image, noise_fraction=2/15):
    noisy_image = image.copy()
    h, w = image.shape[:2]
    patch_height, patch_width = h // 15, w // 15
    num_noisy_patches = int(noise_fraction * (h * w) / (patch_height * patch_width))

    for _ in range(num_noisy_patches):
        x = np.random.randint(0, h - patch_height)
        y = np.random.randint(0, w - patch_width)
        noisy_image[x:x + patch_height, y:y + patch_width] = 128  # Gray color

    return noisy_image

# 读取图像文件，替换为你的图像路径
image_path = './data/m3fd/vi/00000.png'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 添加噪声
noisy_image = add_noise_to_image(image)

output_path = 'noisy_image.png'
noisy_image_bgr = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_path, noisy_image_bgr)
