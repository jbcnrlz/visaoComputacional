import cv2, numpy as np
from sklearn.svm import SVC
def main():
    data = [
        [
            "hog/brain_tumor/Astrocitoma T1/0c14dccd685d7ce330d14fa7a1f53dc756e73aff2f03afc1b09a7efc410f1804_big_gallery.jpeg",
            "hog/brain_tumor/Astrocitoma T1/0fbe78c3db179f94296e3b3d8c05fb_big_gallery.jpeg",
            "hog/brain_tumor/Astrocitoma T1/1b829d504373126112d3a30ca488a6_big_gallery.jpeg",
            "hog/brain_tumor/Astrocitoma T1/1c0238417532d40ffca1c260427b39_big_gallery.jpeg",
            "hog/brain_tumor/Astrocitoma T1/1d0f25e228a7078503a9f0c97ea738c5eed46c3de3dc3215b60c8add9aa77a62_big_gallery.jpeg",
            "hog/brain_tumor/Astrocitoma T1/1ddfe350879837a569a4dc3c4a8aad07b0894e57aff90f0f3e4d80b47ff818dc_big_gallery.jpeg"
        ],[
            "hog/brain_tumor/_NORMAL T1/0a0bc6879f5d5d14c4df229b64b801_big_gallery.jpeg",
            "hog/brain_tumor/_NORMAL T1/0bc0d35606279e59aa78189e824da2_big_gallery.jpeg",
            "hog/brain_tumor/_NORMAL T1/0e66a91abb3c685eb0162351a22385_big_gallery.jpeg",
            "hog/brain_tumor/_NORMAL T1/0f2da85a8272097226e7f352719028_big_gallery.jpeg",
            "hog/brain_tumor/_NORMAL T1/0f66b80311697aa7de23d3b52c736c_big_gallery.jpeg",
            "hog/brain_tumor/_NORMAL T1/0f43646be8dd8f9efff0daae7fed67_big_gallery.jpeg"
        ]
    ]
    test = [['hog/brain_tumor/_NORMAL T1/1a5541b8015fd46f1e71a4a6a14763_big_gallery.jpeg',1],
            ['hog/brain_tumor/Astrocitoma T1/38f5c37719017868f47cd23babddb5_big_gallery.jpg',0]]
    dataset = []
    classes = []
    hog = cv2.HOGDescriptor()
    for cl, d in enumerate(data):
        for b in d:
            img = cv2.imread(b)
            img = cv2.resize(img,(256,256))
            a = hog.compute(img)
            dataset.append(a)
            classes.append(cl)

    dataset = np.array(dataset)    
    clf = SVC()
    clf.fit(dataset,np.array(classes))

    for t in test:
        img = cv2.imread(t[0])
        img = cv2.resize(img,(256,256))
        a = hog.compute(img).reshape(1,-1)
        print(clf.predict(a))
    

if __name__ == '__main__':
    main()