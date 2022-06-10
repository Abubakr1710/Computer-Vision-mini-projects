import cv2
import matplotlib.pyplot as plt
import numpy as np

def screen(img_path, background_path, size):
    img=cv2.imread(img_path)
    bg=cv2.imread(background_path)

    new_img=cv2.resize(img,size)
    nbg=cv2.resize(bg,size)
    up_blue= np.array([255,255, 180])
    l_blue=np.array([113,0,100])
    mask=cv2.inRange(new_img, l_blue, up_blue)
    res=cv2.bitwise_and(new_img, new_img, mask=mask)
    f=new_img-res
    f=np.where(f==0, nbg, f)
    filename='img/whitescreen.png'
    cv2.imwrite(filename, f)
    cv2.imshow('f',f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

screen(img_path='img/my_imgb.jpg',background_path='img/backg.jpg',size=(1280, 720))