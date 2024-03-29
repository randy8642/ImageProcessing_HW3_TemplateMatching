import numpy as np
import cv2
import queue
from multiprocessing.pool import ThreadPool

def change2ConvView(x, y):
    # change view for conv
    view_shape = tuple(np.subtract(x.shape, y.shape) + 1) + y.shape
    strides = x.strides + x.strides
    sub_matrices = np.lib.stride_tricks.as_strided(x, view_shape, strides)

    return sub_matrices

def createSubsampleImgs(originImage, originTemplate):
    sample_imgs = [originImage]
    sample_templates = [originTemplate]

    # 取影像的奇數行與列來縮小影像大小
    for _ in range(3):
        sample_imgs.append(sample_imgs[-1][::2, ::2])
        sample_templates.append(sample_templates[-1][::2, ::2])
    
    return sample_imgs, sample_templates

def deduplicate(location: np.ndarray, similarity: np.ndarray, threshold):
    results_location = []
    results_sim = []

    while len(location) > 0:    
        p = location[0]        

        # 計算p點到其他目標點的距離
        distance = np.sqrt(np.sum((location - p)**2, axis=1))
        # 找出距離過近的點
        nearPoint = np.argwhere(distance < threshold).flatten()

        # 找出這些點之中相似度最高的點
        sim_argmax = np.argmax(similarity[nearPoint])
        # 保留該點
        results_location.append(location[nearPoint][sim_argmax])
        results_sim.append(similarity[nearPoint][sim_argmax])
        # 刪除其他點
        location = np.delete(location, nearPoint, axis=0)
        similarity = np.delete(similarity, nearPoint, axis=0) 
    
    return np.array(results_location), np.array(results_sim)

def templateMatching(img, T):
    # 將圖片依照模板大小重新排列/裁切
    I = change2ConvView(img, T)
    
    # 取得模板的長寬
    template_h, template_w = T.shape
    c = template_h * template_w

    # 計算T(模板)總和與I(圖片區塊)總和
    T_sum = np.einsum('ij->', T)
    I_sum = np.einsum('klij->kl', I)

    # 使用covariant方式計算
    # E(XY) - (x平均 * y平均)
    m = np.einsum('ij,klij->kl', T, I)
    n = T_sum * I_sum / c

    # var(X') = cov(X', X') = E(XX) - (x平均 * x平均)
    I_var = np.einsum('klij->kl', change2ConvView(img**2, T)) - I_sum**2 / c
    T_var = np.einsum('ij->', T**2) - T_sum**2 / c
    b = np.sqrt(T_var * I_var)

    return (m - n) / b

def drawRec(img, loc, sim, t_w, t_h):
    for pt, similarity in zip(loc, sim):
        cv2.rectangle(img, (pt[1], pt[0]), (pt[1] + t_w, pt[0] + t_h), (0, 0, 255), 2)
    for pt, similarity in zip(loc, sim):
        cv2.putText(img, f'center [{pt[1] + t_w//2},{pt[0] + t_h//2}]', (pt[1] + t_w//2 - 50, pt[0] + t_h//2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f'score {similarity:.4}', (pt[1] + t_w//2 - 50, pt[0] + t_h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return img

def convertBGR2GRAY(img):
    res = img[:, :, 2]*0.299 + img[:, :, 1]*0.587 + img[:, :, 0]*0.114
    res = res.astype(np.float32)
    return res


def speedUp_templateMatching(img, templ, threshold):
    # sub sampling
    sample_imgs, sample_templates = createSubsampleImgs(img, templ)

    results_loc = []
    results_sim = []

    def find(now_level, start_point, subimage_w, subimage_h):    

        now_template = sample_templates[now_level]
        now_img = sample_imgs[now_level]

        template_h, template_w = now_template.shape

        # 裁切影像
        h_upper = start_point[0] - 5
        h_lower = start_point[0] + subimage_h + 5
        w_upper = start_point[1] - 5
        w_lower = start_point[1] + subimage_w + 5

        if h_upper < 0:
            h_upper = 0
        if h_lower >= now_img.shape[0]:
            h_lower = now_img.shape[0]
        if w_upper < 0:
            w_upper = 0
        if w_lower >= now_img.shape[1]:
            w_lower = now_img.shape[1]

        cut_img = now_img[h_upper:h_lower, w_upper:w_lower]

        
        # pad
        if (now_template.shape[0] >= cut_img.shape[0]) or (now_template.shape[1] >= cut_img.shape[1]):
            pad = np.array([now_template.shape[0] - cut_img.shape[0], now_template.shape[1] - cut_img.shape[1]], dtype=np.int32) //2 
        
            pad[pad >= 0] += 4
            pad[pad < 0] = 0
        
            padded_img_gray = np.pad(cut_img, [[pad[0], pad[0]], [pad[1], pad[1]]])
        else:
            padded_img_gray = cut_img


        R = templateMatching(padded_img_gray, now_template)

        #     
        if now_level == len(sample_imgs) - 1:
            loc = np.where( R >= threshold)
            points_loc = np.array([[i, j] for i, j in zip(*loc)])
            points_sim = np.array([R[i, j] for i, j in zip(*loc)])
            targetPoint, _ = deduplicate(points_loc, points_sim, sample_imgs[now_level].shape[1]*0.1)
        else:
            loc = np.where( R >= R.max())
            targetPoint = [[loc[0][0], loc[1][0]]]
      
        if len(targetPoint) <= 0:       
            return []    

        #
        if now_level == 0:  
            results_loc.append((targetPoint[0] + np.array([h_upper, w_upper])))      
            results_sim.append(R[(targetPoint[0])[0], (targetPoint[0])[1]])  
            return []    
        #
        targetPoint = targetPoint + np.array([h_upper, w_upper]) + [template_h//2, template_w//2]

        result = []
        for x, y in targetPoint:
            x_init = x - int(now_template.shape[0]*0.5) if x - now_template.shape[0]*0.5 > 0 else 0
            y_init = y - int(now_template.shape[1]*0.5) if y - now_template.shape[1]*0.5 > 0 else 0

            result.append([now_level-1, (np.array([x_init, y_init]))*2, int((now_template.shape[1])*2), int((now_template.shape[0])*2)])

        return result

    pool = ThreadPool(processes=4)

    # init. task
    init_level = len(sample_imgs) - 1
    que = queue.Queue()    

    que.put(pool.apply_async(find, [init_level, np.array([0, 0]), sample_imgs[-1].shape[1], sample_imgs[-1].shape[0]]))
    while not que.empty():
        r = que.get()
        for rr in r.get():
            que.put(pool.apply_async(find, rr))
      
   

    return results_loc, results_sim