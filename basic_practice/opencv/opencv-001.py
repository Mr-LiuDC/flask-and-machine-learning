import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./images/1.jpg')
print(image.shape)
print(image[0, 0])
plt.imshow(image)
plt.show()
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
