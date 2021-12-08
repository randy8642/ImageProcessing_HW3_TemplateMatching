import cv2
import numpy as np
import os
import argparse
from functions import drawRec, convertBGR2GRAY, speedUp_templateMatching
def main(imgPath: str, templatePath: str, threshold: float):
    img = cv2.imread(imgPath)
    template = cv2.imread(templatePath)

    img_gray = convertBGR2GRAY(img)
    template_gray = convertBGR2GRAY(template)

    results_loc, results_sim = speedUp_templateMatching(img_gray, template_gray, threshold)
    
    img_result = drawRec(img.copy(), results_loc, results_sim, *template_gray.shape[::-1])
    
    # cv2.imshow('result', img_result)
    # cv2.waitKey(0)

    imgbaseName = os.path.basename(imgPath).split('.')[0]
    cv2.imwrite(f'./{imgbaseName}.jpg', img_result)

if __name__ == '__main__':   

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='./source/100-1.jpg', type=str)
    parser.add_argument('--template', default='./template/100-Template.jpg', type=str)
    args = parser.parse_args()
    

    imgPath = args.img
    templatePath = args.template
    threshold = 0.6

    
    main(imgPath, templatePath, threshold)
    