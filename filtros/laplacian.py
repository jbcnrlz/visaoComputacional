import cv2
from scipy.ndimage import gaussian_filter

def main():
    imageOri = cv2.imread("D:/PycharmProjects/visaoComputacional/filtros/imagens/lenna.png",cv2.IMREAD_GRAYSCALE)
    image = gaussian_filter(imageOri,sigma=1)
    laplacianoNormal = cv2.Laplacian(imageOri,cv2.CV_64F)
    laplacianNotNormal = cv2.Laplacian(image,cv2.CV_64F)
    cv2.imwrite('filtros/imagens/laplacian.png',laplacianoNormal)
    cv2.imwrite('filtros/imagens/laplacianNotNormal.png',laplacianNotNormal)

if __name__ == '__main__':
    main()