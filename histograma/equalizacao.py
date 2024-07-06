import cv2, matplotlib.pyplot as plt

def main():    
    catImage = cv2.imread('D:/PycharmProjects/visaoComputacional/histograma/images/foto1.jpg',cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([catImage], [0], None, [256], [0, 256])
    catEqualized = cv2.equalizeHist(catImage)
    hist2 = cv2.calcHist([catEqualized], [0], None, [256], [0, 256])
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clImg = clahe.apply(catImage)
    hist3 = cv2.calcHist([clImg], [0], None, [256], [0, 256])

    hist = cv2.normalize(hist,None, 0, 1.0, cv2.NORM_MINMAX)
    hist2 = cv2.normalize(hist2,None, 0, 1.0, cv2.NORM_MINMAX)
    hist3 = cv2.normalize(hist3,None, 0, 1.0, cv2.NORM_MINMAX)

    cv2.imwrite('D:/PycharmProjects/visaoComputacional/histograma/images/foto1_gs.jpg',catImage)
    cv2.imwrite('D:/PycharmProjects/visaoComputacional/histograma/images/foto1_eq.jpg',catEqualized)
    cv2.imwrite('D:/PycharmProjects/visaoComputacional/histograma/images/foto1_clahe.jpg',clImg)

    fig, axs = plt.subplots(3)
    fig.suptitle("Histogramas")    
    axs[0].plot(hist)
    axs[1].plot(hist2)
    axs[2].plot(hist3)
    plt.show()


if __name__ == '__main__':
    main()