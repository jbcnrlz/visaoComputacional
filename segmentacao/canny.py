import cv2, numpy as np

def main():
    imageFile = cv2.imread("D:/PycharmProjects/visaoComputacional/segmentacao/images/placa.jpg",cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(imageFile, (5, 5), 0)
    cv2.imshow("Original", imageFile)
    edges = cv2.Canny(blurred,100,200)
    im_floodfill = edges.copy()
    h, w = edges.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    cv2.imshow("Limiar", im_floodfill_inv)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()