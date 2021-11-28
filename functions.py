import numpy as np
import cv2
from scipy.signal import convolve2d

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

def createSubsampleImgs(originImage, originTemplate):
    sample_imgs = [originImage]
    sample_templates = [originTemplate]

    for _ in range(3):
        sample_imgs.append(sample_imgs[-1][::2, ::2])
        sample_templates.append(sample_templates[-1][::2, ::2])

    # openCV CannyEdge
    # sample_imgs = [cv2.Canny(i, 200, 250) for i in sample_imgs]
    # sample_templates = [cv2.Canny(i, 200, 250) for i in sample_templates]

    # 
    # Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    # Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    # def cal_sobel_amp(i):
    #     G = np.hypot(convolve2d(i, Kx), convolve2d(i, Ky))        
    #     G = (G - G.min()) / (G.max() - G.min()) * 255
    #     G[G < 50] = 0
    #     G[G >= 50] = 255

    #     return G
    

    # sample_imgs = [cal_sobel_amp(i) for i in sample_imgs]
    # sample_templates = [cal_sobel_amp(i) for i in sample_templates]
    
    return sample_imgs, sample_templates

def cannyEdgeDetect(img, threshold):



    return 0