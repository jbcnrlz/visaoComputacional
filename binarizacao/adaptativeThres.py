import cv2

def main():
    imageFile = cv2.imread("D:/PycharmProjects/visaoComputacional/binarizacao/images/moeda.jpg",cv2.IMREAD_GRAYSCALE)
    tImage = cv2.adaptiveThreshold(imageFile,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
    cv2.imshow("Threshold Adaptative", tImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()