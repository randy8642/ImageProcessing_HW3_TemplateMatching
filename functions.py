import numpy as np
import cv2
from scipy.signal import convolve2d
import queue

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
    
    return sample_imgs, sample_templates

def cannyEdgeDetect(img, threshold):



    return 0

def deduplicate(location: np.ndarray, similarity: np.ndarray, threshold):
    results_location = []
    results_sim = []

    while len(location) > 0:    
        p = location[0]        

        distance = np.sqrt(np.sum((location - p)**2, axis=1))
        nearPoint = np.argwhere(distance < threshold).flatten()

        sim_argmax = np.argmax(similarity[nearPoint])
        results_location.append(location[nearPoint][sim_argmax])
        results_sim.append(similarity[nearPoint][sim_argmax])

        location = np.delete(location, nearPoint, axis=0)
        similarity = np.delete(similarity, nearPoint, axis=0) 
    
    return np.array(results_location), np.array(results_sim)

def templateMatching(img, temp):
    sub_matrices = change2ConvView(img, temp)

    #
    h, w = temp.shape
    T_norm = temp - np.mean(temp)  
    I_norm_mean = np.einsum('klij->kl', sub_matrices) / (w*h)

    #
    m = np.einsum('ij,klij->kl', T_norm, sub_matrices)
    n = I_norm_mean * np.sum(T_norm)

    
    T_norm_square_sum = np.sum(T_norm**2)    
    I_norm_square_sum = np.einsum('klij,klij->kl', sub_matrices, sub_matrices) \
                    - 2 * np.einsum('klij->kl', sub_matrices) * I_norm_mean \
                    + (w*h) * I_norm_mean**2

    R = (m - n) / np.sqrt(T_norm_square_sum * I_norm_square_sum)

    return R

def drawRec(img, loc, sim, t_w, t_h):
    for pt, similarity in zip(loc, sim):
        cv2.rectangle(img, (pt[1] - t_w//2, pt[0] - t_h//2), (pt[1] + t_w//2, pt[0] + t_h//2), (0, 0, 255), 2)
    for pt, similarity in zip(loc, sim):
        cv2.putText(img, f'center [{pt[1]},{pt[0]}]', (pt[1] - 50, pt[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f'score {round(similarity, 3)}', (pt[1] - 50, pt[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return img

def convertBGR2GRAY(img):
    res = img[:, :, 2]*0.299 + img[:, :, 1]*0.587 + img[:, :, 0]*0.114
    res = res.astype(np.uint8)
    return res

def speedUp_subsample(img, templ, tm_function, threshold):
    # sub sampling
    sample_imgs, sample_templates = createSubsampleImgs(img, templ)

    # init. task
    init_level = len(sample_imgs) - 1
    que = queue.Queue()
    # [init_level], [start_point], [w], [h]
    que.put([init_level, np.array([0, 0]), sample_imgs[-1].shape[1], sample_imgs[-1].shape[0]])

    results_loc = []
    results_sim = []
    while not que.empty():
        now_level, start_point, subimage_w, subImage_h = que.get()  

        now_template = sample_templates[now_level]
        now_img = sample_imgs[now_level][start_point[0]:start_point[0] + subImage_h, start_point[1]:start_point[1] + subimage_w]

        R = tm_function(now_img, now_template)

        #     
        loc = np.where( R >= threshold)

        points_loc = np.array([[i, j] for i, j in zip(*loc)])
        points_sim = np.array([R[i, j] for i, j in zip(*loc)])
        targetPoint, _ = deduplicate(points_loc, points_sim, sample_imgs[now_level].shape[1]*0.1)
        
        if len(targetPoint) <= 0:       
            continue    
        #
        if now_level == 1:  
            results_loc.append((targetPoint[0] + start_point)*2)      
            results_sim.append(R[(targetPoint[0])[0], (targetPoint[0])[1]])  
            continue    
        #
        targetPoint = targetPoint + start_point
        for x, y in targetPoint:
            h_upper = x - int(now_template.shape[0]*0.5) if x - now_template.shape[0]*0.5 > 0 else 0
            w_upper = y - int(now_template.shape[1]*0.5) if y - now_template.shape[1]*0.5 > 0 else 0

            que.put([now_level-1, np.array([h_upper, w_upper])*2, int(now_template.shape[1]*2), int(now_template.shape[0]*2)])
    
    return results_loc, results_sim