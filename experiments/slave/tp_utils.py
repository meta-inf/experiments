# Monitor classes adapted from tensorpack.callbacks.monitor, presumably authored by Yuxin Wu.

import numpy as np
import tensorflow as tf
from tensorpack.tfutils.summary import create_scalar_summary, create_image_summary


def image_to_nhwc(arr):
    if arr.ndim == 4:
        pass
    elif arr.ndim == 3:
        if arr.shape[-1] in [1, 3, 4]:
            arr = arr[np.newaxis, :]
        else:
            arr = arr[:, :, :, np.newaxis]
    elif arr.ndim == 2:
        arr = arr[np.newaxis, :, :, np.newaxis]
    else:
        raise ValueError("Array of shape {} is not an image!".format(arr.shape))
    return arr

