import cv2
import glob
import os
import numpy as np
from random import randint

data_dir = 'C:/Users/Saurab/Desktop/Tomato' # replace with your data directory

# define functions for data augmentation
def rotate_image(image, angle):
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))

def adjust_brightness(image, brightness_range):
    brightness_offset = int(randint(*brightness_range))
    brightness = np.ones(image.shape, dtype=image.dtype) * brightness_offset
    return cv2.add(image, brightness)

def adjust_contrast(image, contrast_range):
    contrast_alpha = int(randint(*contrast_range)) / 10
    return cv2.addWeighted(image, contrast_alpha, np.zeros_like(image), 0, 0)

def flip_image(image):
    return cv2.flip(image, 1)

# apply data augmentation to each image in every directory
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    if not os.path.isdir(label_dir):
        continue
    images = glob.glob(os.path.join(data_dir, label, "*.jpg"))
    for i, image_path in enumerate(images):
        # read image and apply augmentation
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        rotated = rotate_image(img, randint(-30, 30))
        brightened = adjust_brightness(rotated, [0, 30])
        adjusted = adjust_contrast(brightened, [5, 15])
        flipped = flip_image(adjusted)

        # save augmented images
        save_dir = os.path.join('Dataset/Plant', label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format(i)), img)
        cv2.imwrite(os.path.join(save_dir, '{}_rotated.jpg'.format(i)), rotated)
        cv2.imwrite(os.path.join(save_dir, '{}.brightened.jpg'.format(i)), brightened)
        cv2.imwrite(os.path.join(save_dir, '{}.adjusted.jpg'.format(i)), adjusted)
        cv2.imwrite(os.path.join(save_dir, '{}.flipped.jpg'.format(i)), flipped)
