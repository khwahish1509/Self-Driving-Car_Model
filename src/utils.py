import os
import argparse
import pandas as pd
import numpy as np
import cv2


# Utility to merge multiple driving_logs and balance steering distribution


def merge_csvs(raw_dir, out_csv):
rows = []
for root, dirs, files in os.walk(raw_dir):
for f in files:
if f.endswith('.csv'):
p = os.path.join(root, f)
df = pd.read_csv(p, header=None)
# expect typical Udacity