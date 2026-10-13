"""The data set of Chapters 5 to 8 and 13: the 1,797 handwritten digits that ship with scikit-learn.

Each image is 8 x 8 pixels with values from 0 to 16. The data are in the scikit-learn package, so the
book needs no network. The split is stratified and fixed, so every chapter sees the same 1,347
training images and the same 450 test images.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load(scale: bool = True, images: bool = False):
    """Return x_train, x_test, y_train, y_test as float32 arrays and int labels.

    scale=True divides the pixels by 16, so that they lie in [0, 1].
    images=True returns arrays of shape (n, 8, 8, 1) instead of (n, 64).
    """
    data = load_digits()
    x = data.data.astype("float32")
    if scale:
        x = x / 16.0
    if images:
        x = x.reshape(-1, 8, 8, 1)
    return train_test_split(x, data.target, test_size=0.25, random_state=0, stratify=data.target)
