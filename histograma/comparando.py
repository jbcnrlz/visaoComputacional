import cv2, matplotlib.pyplot as plt, numpy as np

def main():
    imagens = [
        'D:/PycharmProjects/visaoComputacional/histograma/images/foto1.jpg',
        'D:/PycharmProjects/visaoComputacional/histograma/images/foto2.jpg',
        'D:/PycharmProjects/visaoComputacional/histograma/images/foto3.jpg',
        'D:/PycharmProjects/visaoComputacional/histograma/images/foto4.jpg',
        'D:/PycharmProjects/visaoComputacional/histograma/images/foto5.jpg',
        'D:/PycharmProjects/visaoComputacional/histograma/images/foto6.jpg',
        'D:/PycharmProjects/visaoComputacional/histograma/images/gatinho.png'        
    ]
    results = {}
    dataFrameResult = np.zeros((len(imagens),len(imagens)))
    for idxGal, i in enumerate(imagens):
        results[i] = []
        imageBank = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clImg = clahe.apply(imageBank)
        clImg = cv2.calcHist([clImg], [0], None, [256], [0, 256])
        clImg = cv2.normalize(clImg,None, 0, 1.0, cv2.NORM_MINMAX)
        for idxProbe, j in enumerate(imagens):
            imageProbe = cv2.imread(j,cv2.IMREAD_GRAYSCALE)
            claheP = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clImgP = claheP.apply(imageProbe)
            clImgP = cv2.calcHist([clImgP], [0], None, [256], [0, 256])
            clImgP = cv2.normalize(clImgP,None, 0, 1.0, cv2.NORM_MINMAX)
            dataFrameResult[idxGal][idxProbe] = cv2.compareHist(clImg,clImgP,1)
    fulImage = None
    for idx in range(len(imagens)):
        idxsVals = dataFrameResult[idx].argsort()
        currImg = cv2.imread(imagens[idx])
        currImg = cv2.resize(currImg,(250,250))
        matchedImg = cv2.imread(imagens[idxsVals[1]])
        matchedImg = cv2.resize(matchedImg,(250,250))
        fImage = np.concatenate((currImg,matchedImg),axis=0)
        if fulImage is not None:
            fulImage = np.concatenate((fulImage,fImage),axis=1)
        else:
            fulImage = fImage
    cv2.imshow("Janela da Imagem",fulImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()