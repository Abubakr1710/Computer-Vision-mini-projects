import cv2
import matplotlib.pyplot as plt
img=cv2.imread('img/minion.jpg')
m = cv2.putText(img,'SPACE and COMMA when I forgot them 500 line of code', (95,15), cv2.FONT_HERSHEY_COMPLEX,0.5, (255,255,255))
path='img/meme.png'
cv2.imwrite(path,m)
meme = cv2.imread('img/meme.png')
rgb_meme= meme.copy()
rgb_meme=cv2.cvtColor(rgb_meme, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,15))
plt.imshow(rgb_meme)
plt.show()


