import cv2
import numpy as np


def crop_image(img, top=60, bottom=20):
    return img[top:img.shape[0]-bottom, :]


def convert_to_yuv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


def resize(img, new_size=(200,66)):
    return cv2.resize(img, new_size)


def normalize(img):
# input img expected uint8 0..255
    return img.astype(np.float32)/255.0 - 0.5


def preprocess(img, do_crop=True):
    if do_crop:
        img = crop_image(img)
    img = convert_to_yuv(img)
    img = resize(img)
    img = normalize(img)
    return img