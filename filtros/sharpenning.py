import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def main():
    imageFile = cv2.imread(
        "D:/PycharmProjects/visaoComputacional/filtros/imagens/lenna.png",
        cv2.IMREAD_GRAYSCALE
    )
    newImage = gaussian_filter(imageFile,sigma=1.5)
    details = imageFile - newImage
    
    sharp = imageFile + details
    cv2.imwrite("filtros/imagens/original.png",imageFile)
    cv2.imwrite("filtros/imagens/details.png",details)
    cv2.imwrite("filtros/imagens/sharpfilter.png",sharp)


if __name__ == '__main__':
    main()