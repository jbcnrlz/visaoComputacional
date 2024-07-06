import cv2, matplotlib as plt

def main():    
    catImage = cv2.imread('D:/PycharmProjects/visaoComputacional/introducao/images/gatinho.png',cv2.IMREAD_GRAYSCALE)

    #inicioRetangulo = (20,10)
    #fimRetangulo = (100,90)
    #cor = (0,0,255)
    #grossura = 5
    #catImage = cv2.rectangle(catImage,inicioRetangulo,fimRetangulo,cor,grossura)

    cv2.imshow("Janela da imagem",catImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('D:/PycharmProjects/visaoComputacional/introducao/images/gatinhoGray.jpg',catImage)


if __name__ == '__main__':
    main()