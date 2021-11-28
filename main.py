import cv2
import numpy as np
import queue
from functions import change2ConvView, createSubsampleImgs

imgName = 'Die1.tif'
templateName = 'Die-Template.tif'

img = cv2.imread(f'./source/{imgName}')
template = cv2.imread(f'./template/{templateName}')

img_gray = img[:, :, 2]*0.299 + img[:, :, 1]*0.587 + img[:, :, 0]*0.114
img_gray = img_gray.astype(np.uint8)

template_gray = template[:, :, 2]*0.299 + template[:, :, 1]*0.587 + template[:, :, 0]*0.114
template_gray = template_gray.astype(np.uint8)

# sub sampling
sample_imgs, sample_templates = createSubsampleImgs(img_gray, template_gray)

# init. task
init_level = len(sample_imgs) - 1

que = queue.Queue()

# [init_level], [start_point], [w], [h]
que.put([init_level, np.array([0, 0]), sample_imgs[-1].shape[1], sample_imgs[-1].shape[0]])

results = []
while True:
    if que.empty():
        break    
    
    now_level, start_point, subimage_w, subImage_h = que.get()  

    now_template = sample_templates[now_level]
    now_img = sample_imgs[now_level][start_point[0]:start_point[0] + subImage_h, start_point[1]:start_point[1] + subimage_w]


    sub_matrices = change2ConvView(now_img, now_template)

    #
    template_h, template_w = now_template.shape
    T_norm = now_template - np.mean(now_template)  
    I_norm_mean = np.einsum('klij->kl', sub_matrices) / (template_w*template_h)

    #
    m = np.einsum('ij,klij->kl', T_norm, sub_matrices)
    n = I_norm_mean * np.sum(T_norm)

    
    T_norm_square_sum = np.sum(T_norm**2)    
    I_norm_square_sum = np.einsum('klij,klij->kl', sub_matrices, sub_matrices) \
                    - 2 * np.einsum('klij->kl', sub_matrices) * I_norm_mean \
                    + (template_w*template_h) * I_norm_mean**2

    R = (m - n) / np.sqrt(T_norm_square_sum * I_norm_square_sum)

    # 
    threshold = 0.65
    loc = np.where( R >= threshold)
  
    
    #
    targetPoint = []
    for n, (x, y) in enumerate(zip(*loc)):
        if n == 0:
            targetPoint.append([x, y])
        else:
            dis = [np.sqrt((x-i)**2 + (y-j)**2) for i, j in targetPoint]
            dis_sort = np.argsort(dis)
            sim = [R[i, j] for i, j in targetPoint]
                        
            if dis[dis_sort[0]] > sample_imgs[now_level].shape[1]*0.1:
                targetPoint.append([x, y])
            else:
                if R[x, y] > sim[dis_sort[0]]:
                    del targetPoint[dis_sort[0]]
                    targetPoint.append([x, y])
            

    
    if len(targetPoint) <= 0:       
        continue

    targetPoint = np.array(targetPoint) + start_point
    
    #
    if now_level == 1:          
        results.append([(targetPoint[0]*2).tolist(), R[(targetPoint[0]-start_point)[0], (targetPoint[0]-start_point)[1]]])
    else:
        for x, y in targetPoint:
            h_upper = x - int(now_template.shape[0]*0.5) if x - now_template.shape[0]*0.5 > 0 else 0
            w_upper = y - int(now_template.shape[1]*0.5) if y - now_template.shape[1]*0.5 > 0 else 0

            que.put([now_level-1, np.array([h_upper, w_upper])*2, int(now_template.shape[1]*2), int(now_template.shape[0]*2)])

# 移除重複目標框
deduplicate_results = []
for n, ((x, y), sim) in enumerate(results):
    if n == 0:
        deduplicate_results.append([[x, y], sim])
    else:
        dis = [np.sqrt((x-i)**2 + (y-j)**2) for (i, j), s in deduplicate_results]
        dis_sort = np.argsort(dis)
                    
        if dis[dis_sort[0]] > sample_imgs[now_level].shape[1]*0.1:
            deduplicate_results.append([[x, y], sim])
        else:
            if sim > deduplicate_results[dis_sort[0]][1]:
                del deduplicate_results[dis_sort[0]]
                deduplicate_results.append([[x, y], sim])

#
img_res = img.copy()
w, h = template_gray.shape[::-1]

for pt, similarity in deduplicate_results:
    cv2.rectangle(img_res, (pt[1] - w//2, pt[0] - h//2), (pt[1] + w//2, pt[0] + h//2), (0, 0, 255), 2)
for pt, similarity in deduplicate_results:
    cv2.putText(img_res, f'center [{pt[1]},{pt[0]}]', (pt[1] - 50, pt[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img_res, f'score {round(similarity, 3)}', (pt[1] - 50, pt[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

# cv2.imshow('result', img_res)
# cv2.waitKey(0)

imgbaseName = imgName.split('.')[0]
cv2.imwrite(f'./result/{imgbaseName}.jpg', img_res)