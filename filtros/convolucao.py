import cv2
import numpy as np
from scipy import signal

def main():
    imageFile = cv2.imread("D:/PycharmProjects/visaoComputacional/filtros/imagens/Noise_salt_and_pepper.png",cv2.IMREAD_GRAYSCALE)
    print(imageFile)
    filterWind = np.ones((3,3)) / 9
    newImage = signal.convolve2d(imageFile,filterWind)
    cv2.imwrite("filtros/imagens/convolved.png",newImage)


if __name__ == '__main__':
    main()