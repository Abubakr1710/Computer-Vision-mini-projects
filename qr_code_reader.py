from fileinput import filename
import cv2
from cv2 import threshold
import matplotlib.pyplot as plt

qr_code = cv2.imread('img/qrcode.png')
print(type(qr_code))
decoder=cv2.QRCodeDetector()
data, point, output3 = decoder.detectAndDecode(qr_code)
if point is not None:
    print('Link is:',data)

img= qr_code.copy()
gray_img=img.copy()
gray_img=cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
retval, thr = cv2.threshold(gray_img,200,255, cv2.THRESH_BINARY_INV)
contours, hr= cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
sorted_counters=sorted(contours, key=cv2.contourArea,reverse=True)
copy_img=img.copy()
x,y,w,h = cv2.boundingRect(sorted_counters[0])
rect=cv2.rectangle(copy_img, (x,y), (x+w,y+h), (0,255,0), 2)
fname='img/rect_qr.png'
cv2.imwrite(fname, rect)
plt.figure(figsize=(20,15))
plt.imshow(rect)
plt.show()