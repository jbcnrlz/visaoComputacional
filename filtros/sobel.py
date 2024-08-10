import cv2
import numpy as np

def main():
    imageFile = cv2.imread("D:/PycharmProjects/visaoComputacional/filtros/imagens/lenna.png",cv2.IMREAD_GRAYSCALE)
    sobel_x = cv2.Sobel(imageFile,cv2.CV_64F,1,0,ksize=3)
    sobel_y = cv2.Sobel(imageFile,cv2.CV_64F,0,1,ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    cv2.imwrite("filtros/imagens/grad_x.png",sobel_x)
    cv2.imwrite("filtros/imagens/grad_y.png",sobel_y)
    cv2.imwrite("filtros/imagens/edges.png",magnitude)


if __name__ == '__main__':
    main()