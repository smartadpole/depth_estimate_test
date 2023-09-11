import numpy as np
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

def np2float(x, t=True, bgr=False):
    if len(x.shape) == 2:
        x = x[..., None]
    if bgr:
        x = x[..., [2, 1, 0]]
    if t:
        x = np.transpose(x, (2, 0, 1))
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255
    return x

def preprocess_hit(image):
    image = np2float(image)
    image = np.expand_dims(image, axis=0).astype(np.float32)
    image = image * 2 - 1
    return image

def preprocess_madnet(image):
    image = np2float(image, bgr=True)
    image = image[None, ...]
    return image