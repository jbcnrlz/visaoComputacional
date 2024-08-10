import cv2
import numpy as np

def main():
    imageFile = cv2.imread("D:/PycharmProjects/visaoComputacional/filtros/imagens/Noise_salt_and_pepper.png",cv2.IMREAD_GRAYSCALE)
    print(imageFile)
    newImage = np.zeros(imageFile.shape)
    for i in range(newImage.shape[0]-2):
        for j in range(newImage.shape[1]-2):
            currWindow = imageFile[i:i+3,j:j+3]
            newImage[(3 // 2) + i,(3 // 2) + j] = np.median(currWindow)

    cv2.imwrite("filtros/imagens/median.png",newImage)


if __name__ == '__main__':
    main()