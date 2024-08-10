import cv2

def main():
    imageFile = cv2.imread("D:/PycharmProjects/visaoComputacional/segmentacao/images/placa.jpg",cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(imageFile, (7, 7), 0)
    cv2.imshow("Original", imageFile)
    (T, threshInv) = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("Limiar", threshInv)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()