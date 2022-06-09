import cv2
import matplotlib.pyplot as plt
img=cv2.imread('img/minion.jpg')

m = cv2.putText(img,'SPACE and COMMA when I forgot them 500 line of code', org=(0,20), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=0.6,color=(255,255,255), thickness=1)
mdown= cv2.putText(m,'Do not forget them', org=(350, 345), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=0.7,color=(255,255,255), thickness=1)
path='img/meme.png'
cv2.imwrite(path,m)
meme = cv2.imread('img/meme.png')
rgb_meme= meme.copy()
rgb_meme=cv2.cvtColor(rgb_meme, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,15))
plt.imshow(rgb_meme)
plt.show()


