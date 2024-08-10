import cv2
import numpy as np

def main():
    galColor = cv2.imread("D:/PycharmProjects/visaoComputacional/filtros/imagens/gallery.jpg")
    gal = cv2.cvtColor(galColor,cv2.COLOR_BGR2GRAY)
    probe = cv2.imread("D:/PycharmProjects/visaoComputacional/filtros/imagens/pot.jpg",cv2.IMREAD_GRAYSCALE)
    match = cv2.matchTemplate(gal, probe,cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(match)
    (startX, startY) = maxLoc
    endX = startX + probe.shape[1]
    endY = startY + probe.shape[0]
    cv2.rectangle(galColor, (startX, startY), (endX, endY), (255, 0, 0), 3)
    cv2.imshow("Output", galColor)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()