import cv2
import matplotlib.pyplot as plt

qr_code = cv2.imread('img/qrcode.png')
print(type(qr_code))
decoder=cv2.QRCodeDetector()
data, point, _ = decoder.detectAndDecode(qr_code)
if point is not None:
    print('Link is:',data)
plt.imshow(qr_code)
plt.show()
