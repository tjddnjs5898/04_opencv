# 평균 필터를 생상하여 블러 적용 (blur_avg_kernel.py)

import cv2
import numpy as np

img = cv2.imread('../img/gaussian_noise.jpg')

kernel = np.ones((5,5))/5**2

blured = cv2.filter2D(img, -1, kernel)

cv2.imshow('origin', img)
cv2.imshow('avrg blur', blured)
cv2.waitKey()
cv2.destroyAllWindows()
