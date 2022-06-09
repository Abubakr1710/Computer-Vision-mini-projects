import cv2
import matplotlib.pyplot as plt
from torch import qr

d = cv2.imread('img/qrcode.png')
plt.imshow(d)
plt.show()
code_reader=cv2.QRCodeDetector()
data, points, _ = code_reader.detectAndDecode(d)
if points is not None:
    print('Link is:',data)