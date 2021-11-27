import numpy as np


def change2ConvView(x, y):
    # pad
    pad = np.array(y.shape) // 2
    padded_img_gray = np.ones([x.shape[0] + pad[0]*2, x.shape[1] + pad[1]*2]) * 0
    padded_img_gray[pad[0]:-pad[0], pad[1]:-pad[1]] = x

    # change view for conv
    view_shape = tuple(np.subtract(padded_img_gray.shape, y.shape) + 1) + y.shape
    strides = padded_img_gray.strides + padded_img_gray.strides
    sub_matrices = np.lib.stride_tricks.as_strided(padded_img_gray, view_shape, strides)

    return sub_matrices