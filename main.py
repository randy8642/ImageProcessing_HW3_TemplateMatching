import cv2
import numpy as np
import argparse
from functions import deduplicate, templateMatching, drawRec, convertBGR2GRAY, speedUp_subsample

def main(imgPath: str, templatePath: str, threshold: float):
    img = cv2.imread(imgPath)
    template = cv2.imread(templatePath)

    img_gray = convertBGR2GRAY(img)
    template_gray = convertBGR2GRAY(template)

    results_loc, results_sim = speedUp_subsample(img_gray, template_gray, templateMatching, threshold)

    # 移除重複目標框
    res_loc, res_sim = deduplicate(np.array(results_loc), np.array(results_sim), img_gray.shape[1]*0.1)

    #
    img_result = drawRec(img.copy(), res_loc, res_sim, *template_gray.shape[::-1])

    # cv2.imshow('result', img_result)
    # cv2.waitKey(0)

    # imgbaseName = imgName.split('.')[0]
    # cv2.imwrite(f'./result/{imgbaseName}.jpg', img_res)

if __name__ == '__main__':   
    imgPath = './source/100-4.jpg'
    templatePath = './template/100-Template.jpg'
    threshold = 0.6

    import time
    st = time.time()

    main(imgPath, templatePath, threshold)

    print(time.time() - st)