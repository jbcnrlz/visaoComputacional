import cv2

def main():
    img = cv2.imread('hog/imagem/povo.jpg')

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    locations, confidence = hog.detectMultiScale(img)

    for (x, y, w, h) in locations:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)

    cv2.imshow('Povo', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()