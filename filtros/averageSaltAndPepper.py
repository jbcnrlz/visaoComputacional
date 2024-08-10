import cv2
import numpy as np

def main():
    imageFile = cv2.imread("D:/PycharmProjects/visaoComputacional/filtros/imagens/Noise_salt_and_pepper.png",cv2.IMREAD_GRAYSCALE)
    print(imageFile)
    newImage = np.zeros(imageFile.shape)
    filterWind = np.ones((3,3))
    filterTypes = [9,16]
    for ft in filterTypes:
        for i in range(newImage.shape[0]-2):
            for j in range(newImage.shape[1]-2):
                currWindow = imageFile[i:i+filterWind.shape[0],j:j+filterWind.shape[1]]
                appFilter = currWindow * filterWind
                newImage[(filterWind.shape[0] // 2) + i,(filterWind.shape[0] // 2) + j] = np.sum(appFilter) / ft

        cv2.imwrite("filtros/imagens/%s.png" % ('gaussian' if ft == 16 else 'average'),newImage)


if __name__ == '__main__':
    main()