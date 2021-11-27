import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import itertools
from scipy.signal import convolve2d, correlate2d

img = cv2.imread('./source/100-4.jpg')
template = cv2.imread('./template/100-Template.jpg')

img = img[:, :, 2]*0.299 + img[:, :, 1]*0.587 + img[:, :, 0]*0.114
img = img.astype(np.uint8)

template = template[:, :, 2]*0.299 + template[:, :, 1]*0.587 + template[:, :, 0]*0.114
template = template.astype(np.uint8)

def matching(simg, stempl):
    dst = cv2.Canny(stempl, 200, 1)
    featureMap = dst > 0

    # dst = cv2.cornerHarris(stempl, 2, 3, 0.04)
    # a = (dst - dst.min()) / (dst.max() - dst.min()) * 255
    # a = np.abs(a)
    # featureMap = a < 140


    # pad
    pad = np.array(stempl.shape) // 2
    padded_x = np.ones([simg.shape[0] + pad[0]*2, simg.shape[1] + pad[1]*2]) * 0
    padded_x[pad[0]:-pad[0], pad[1]:-pad[1]] = simg

    simg = padded_x
    
    # sobel 
    Sx = np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])
    Sy = np.array([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]])

    # template gradient
    d = np.empty([*stempl.shape, 2])
    d[:, :, 0] = convolve2d(stempl, Sx, mode='same', boundary='fill', fillvalue=0)
    d[:, :, 1] = convolve2d(stempl, Sy, mode='same', boundary='fill', fillvalue=0)

    # feature position
    p = np.empty([*stempl.shape, 2], dtype=np.int32)    
    for x, y in itertools.product(range(stempl.shape[0]), range(stempl.shape[1])):
        p[x, y, :] = [x, y]
    
    # image grsdient
    e = np.empty([*simg.shape, 2])
    e[:, :, 0] = convolve2d(simg, Sx, mode='same', boundary='fill', fillvalue=0)
    e[:, :, 1] = convolve2d(simg, Sy, mode='same', boundary='fill', fillvalue=0)

    d = d[featureMap]
    p = p[featureMap]

    res = np.zeros(simg.shape)
    for qx, qy in itertools.product(range(simg.shape[0] - stempl.shape[0]), range(simg.shape[1] - stempl.shape[1])):
        
        ps = p + [qx, qy]
        e_part = e[ps[:, 0], ps[:, 1]]
        res[qx, qy] = np.mean(np.nan_to_num((np.sum(e_part * d, axis=-1)) / np.sqrt((d[:, 0]**2 + d[:, 1]**2) * (e_part[:, 0]**2 + e_part[:, 1]**2))))
    
    return res[:-stempl.shape[0], :-stempl.shape[1]]

def search(init_position, w, h, level:int):
    
    if level < 0:
        return []    

   
    # matching
    h_upper = init_position[0] + h
    h_lower = init_position[0]
    w_upper = init_position[1] + w
    w_lower = init_position[1]       
    
    conv_img = matching(imgs[level][h_lower:h_upper, w_lower:w_upper], templates[level])

    

    # find points larger than threshold
    threshold = 0.8
    norm_result = (conv_img - conv_img.min()) / (conv_img.max() - conv_img.min())
    loc = np.where( norm_result >= threshold)
    
    # remove same points
    target_position = []
    for n, (x, y) in enumerate(zip(*loc)):
        if n == 0:
            target_position.append([x, y])
            continue
        
        store = True
        for point in target_position:
            distance = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
            if distance < imgs[level].shape[0] * 0.15:
                store = False
                break
        
        if store:        
            target_position.append([x, y])
    

    if len(target_position) == 0:
        return []

    target_position = np.array(target_position) - np.array(templates[level].shape)//2
   


    # upsample
    results = []
    for search_center in target_position:
        search_center = np.array(search_center)
        if level == 0:            
            return [search_center + init_position]
        else:
            result = search((search_center + init_position) * 2, templates[level-1].shape[1], templates[level-1].shape[0] , level=level-1)
            results.extend(result)
    
    return results

# copys
imgs = []
imgs.append(img)

templates = []
templates.append(template)

n_subsample = 3

for i in range(n_subsample):
    imgs.append(imgs[-1][::2, ::2])
    templates.append(templates[-1][::2, ::2])

tar = search(np.array([0, 0]), imgs[-1].shape[1], imgs[-1].shape[0], level=len(imgs)-1)


img_res = imgs[0].copy()
for pt in tar:  
    cv2.rectangle(img_res, pt[::-1], (pt[1] + templates[0].shape[1], pt[0] + templates[0].shape[0]), (0, 0, 255), 2)
plt.figure(figsize=(10, 10))
plt.imshow(img_res, cmap='gray')
plt.show()