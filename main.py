import cv2
import numpy as np
import os
import argparse
import time
from functions import templateMatching, drawRec, convertBGR2GRAY, speedUp_subsample

def main(imgPath: str, templatePath: str, threshold: float):
    img = cv2.imread(imgPath)
    template = cv2.imread(templatePath)

    img_gray = convertBGR2GRAY(img)
    template_gray = convertBGR2GRAY(template)

    results_loc, results_sim = speedUp_subsample(img_gray, template_gray, templateMatching, threshold)
    
    # img_result = drawRec(img.copy(), results_loc, results_sim, *template_gray.shape[::-1])
    
    # cv2.imshow('result', img_result)
    # cv2.waitKey(0)

    # imgbaseName = os.path.basename(imgPath).split('.')[0]
    # cv2.imwrite(f'./result/{imgbaseName}.jpg', img_result)

if __name__ == '__main__':   
    imgPath = './source/100-4.jpg'
    templatePath = './template/100-Template.jpg'
    threshold = 0.6

    st = time.time()
    main(imgPath, templatePath, threshold)
    print(time.time() - st)

    st = time.time()
    img = cv2.imread(imgPath)
    template = cv2.imread(templatePath)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)
    print(time.time() - st)