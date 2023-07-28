import cv2
import numpy as np
from torchvision import transforms

def unnormalize(img_base):
    aug_mean = np.array([0.485, 0.456, 0.406])
    aug_std = np.array([0.229, 0.224, 0.225])
    unnormalize = transforms.Normalize((-aug_mean / aug_std).tolist(), (1.0 / aug_std).tolist())
    img_unnorm = unnormalize(img_base)

    return img_unnorm

def resize_image(image, new_height):
    aspect_ratio = image.shape[1] / image.shape[0]
    new_width = int(new_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

