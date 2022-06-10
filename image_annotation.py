import cv2
import matplotlib.pyplot as plt
img=cv2.imread('img/cat_human.jpg')
cimg=img.copy()
cimg=cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
cv2.rectangle(cimg,(170,50), (1030,1300), (0,255,0),4)
cv2.rectangle(cimg, (170,50),(350,0), (0,255,0), cv2.FILLED)
cv2.putText(cimg,'Human', (190,30),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
cv2.rectangle(cimg,(1650,450), (1950,1150), (255,0,0),4)
cv2.rectangle(cimg, (1650,400), (1750,450), (255,0,0), cv2.FILLED)
cv2.putText(cimg,'Cat',(1670,430), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0))
filename='img/annotated.png'
bgr_cimg=cimg.copy()
bgr_cimg=cv2.cvtColor(bgr_cimg, cv2.COLOR_RGB2BGR)
cv2.imwrite(filename, bgr_cimg)
plt.figure(figsize=(20,15))
plt.imshow(cimg)
plt.show()