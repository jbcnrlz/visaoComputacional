import cv2, matplotlib.pyplot as plt

def main():
    imageFile = cv2.imread("D:/PycharmProjects/visaoComputacional/binarizacao/images/moeda.jpg",cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(imageFile, (7, 7), 0)

    (T, threshInv) = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow("Threshold", threshInv)
    print("[INFO] otsu's thresholding value: {}".format(T))

    masked = cv2.bitwise_and(imageFile, imageFile, mask=threshInv)
    cv2.imshow("Output", masked)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()