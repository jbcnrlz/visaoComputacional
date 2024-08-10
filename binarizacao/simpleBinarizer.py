import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def main():
    imageFile = cv2.imread("D:/PycharmProjects/visaoComputacional/binarizacao/images/moeda.jpg",cv2.IMREAD_GRAYSCALE)
    (T, tImage) = cv2.threshold(imageFile,200,255,cv2.THRESH_BINARY_INV)
    cv2.imshow("Threshold Binary", tImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()