import cv2
import numpy as np
import pandas as pd
import time
from functions import convertBGR2GRAY, speedUp_subsample


results = []

templatePath = './template/100-Template.jpg'
threshold = 0.6

for _ in range(10):
    for i in range(1, 5, 1):
        imgPath = f'./source/100-{i}.jpg'        

        

        ####################################################
        st = time.time()
        #
        img = cv2.imread(imgPath)
        template = cv2.imread(templatePath)

        img_gray = convertBGR2GRAY(img)
        template_gray = convertBGR2GRAY(template)

        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= threshold)
        
        #
        costTime = time.time() - st
        results.append(['OpenCV function', f'100-{i}.jpg', costTime])

        ####################################################
        st = time.time()
        #
        img = cv2.imread(imgPath)
        template = cv2.imread(templatePath)

        img_gray = convertBGR2GRAY(img)
        template_gray = convertBGR2GRAY(template)

        results_loc, results_sim = speedUp_subsample(img_gray, template_gray, threshold)
        
        #
        costTime = time.time() - st
        results.append(['Self-developed', f'100-{i}.jpg', costTime])

results = pd.DataFrame(results, columns=['method', 'image', 'costTime'])
# results.to_csv('./result/benchmark.csv', index=False, header=True, mode='w')

print(results.groupby(['method']).mean().to_markdown())