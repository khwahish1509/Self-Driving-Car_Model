import cv2
import numpy as np
import random


def random_flip(img, steering):
    if random.random() < 0.5:
        return cv2.flip(img, 1), -steering
    return img, steering


def random_brightness(img):
    # img in BGR or YUV; assume BGR input for brightness adjustment
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ratio = 0.4 + np.random.uniform()
    hsv[:,:,2] = np.clip(hsv[:,:,2]*ratio, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_translate(img, steering, range_x=50, range_y=10):
    tx = range_x * (np.random.rand() - 0.5) * 2
    ty = range_y * (np.random.rand() - 0.5) * 2
    steering += tx * 0.002
    M = np.float32([[1,0,tx],[0,1,ty]])
    rows, cols = img.shape[:2]
    img = cv2.warpAffine(img, M, (cols, rows))
    return img, steering


def random_shadow(image):
# cast a random shadow
    top_x, bottom_x = np.random.randint(0, image.shape[1], 2)
    mask = np.zeros_like(image[:,:,1])
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    mask[((X_m - top_x) * (bottom_x - top_x) >= 0)] = 1
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(0.2, 0.7)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hls[:,:,1][cond] = hls[:,:,1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)


def random_augment(img, steering):
    # apply augmentations with probabilities
    if np.random.rand() < 0.5:
        img = random_brightness(img)
    if np.random.rand() < 0.5:
        img = random_shadow(img)
    if np.random.rand() < 0.5:
        img, steering = random_translate(img, steering)
    img, steering = random_flip(img, steering)
    return img, steering