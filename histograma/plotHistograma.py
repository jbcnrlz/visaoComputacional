import cv2, matplotlib.pyplot as plt

def main():    
    catImage = cv2.imread('D:/PycharmProjects/visaoComputacional/histograma/images/foto1.jpg')
    hist1 = cv2.calcHist(catImage, [0], None, [256], [0, 256])
    hist2 = cv2.calcHist(catImage, [1], None, [256], [0, 256])
    hist3 = cv2.calcHist(catImage, [2], None, [256], [0, 256])
    
    
    plt.title('Histograma P&B')
    plt.xlabel("Intensidade")
    plt.ylabel("Quantidade de Pixels")
    plt.plot(hist1,c='blue')
    plt.plot(hist2,c='green')
    plt.plot(hist3,c='red')
    plt.xlim([0, 256])
    plt.show()


if __name__ == '__main__':
    main()