from __future__ import print_function
from __future__ import absolute_import
import errno
import os
import scipy.misc
import numpy as np

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def merge(images, size):
    if len(images.shape) == 2:
        px = np.sqrt(images.shape[1]).astype(np.int32)
        images = images.reshape((images.shape[0],px,px))
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        if len(image.shape) == 2:
            img[j*h:j*h+h, i*w:i*w+w] = image
        else:
            img[j*h:j*h+h, i*w:i*w+w,:] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))
