import cv2
import numpy as np
import sklearn
from src.preprocess import preprocess
from src.augment import random_augment


class DataGenerator:
    def __init__(self, samples, batch_size=32, is_training=True):
        self.samples = samples
        self.batch_size = batch_size
        self.is_training = is_training


    def __len__(self):
        return int(np.ceil(len(self.samples) / float(self.batch_size)))


    def __iter__(self):
        return self.generator()


    def generator(self):
        num_samples = len(self.samples)
        while True:
                sklearn.utils.shuffle(self.samples)
                for offset in range(0, num_samples, self.batch_size):
                    batch_samples = self.samples[offset:offset + self.batch_size]
                    images = []
                    angles = []
                    for batch_sample in batch_samples:
                        img_path = batch_sample[0]
                        steering = float(batch_sample[1])
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        if self.is_training:
                            img_aug, steering_aug = random_augment(img, steering)
                            img_p = preprocess(img_aug)
                            images.append(img_p)
                            angles.append(steering_aug)
                        else:
                            img_p = preprocess(img)
                            images.append(img_p)
                            angles.append(steering)
                        X = np.array(images)
                        y = np.array(angles)
                        yield sklearn.utils.shuffle(X, y)