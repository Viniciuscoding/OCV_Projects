import numpy as np
import matplotlib.pyplot as plt
import cv2


import matplotlib
matplotlib.rcParams['figure.figsize'] = (18.0, 12.0)
matplotlib.rcParams['image.interpolation'] = 'bilinear'


# Supress warnings
import warnings
warnings.filterwarnings("ignore")

# Read the image. Notice the image is read in BGR by OpenCV.
img = cv2.imread("images/Coins.png")

# Display the image. By inverting the image you get RGB.
plt.imshow(img[:,:,::-1])
plt.show()

# Conversion formula
# BGR to Gray formula -> 0.299*R + 0.587*G + 0.114*B 

# Reference: 
# https://docs.opencv.org/4.1.0/de/d25/imgproc_color_conversions.html 


def convertBGRtoGray(image):
    # Get the image channels
    imgB, imgG, imgR = cv2.split(image)
    
    # Use the formula. Round the results and change to uint8 to display the image
    gray = np.round(0.299*imgR).astype("uint8") + np.round(0.587*imgG).astype("uint8") + np.round(0.114*imgB).astype("uint8") 
    
    return gray

# Image manually changed to gray color
gray = convertBGRtoGray(img)

# OpenCV's `cvtColor` result as a verifier
gray_cv = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Display the manual results in comparison with OpenCV results
plt.figure(figsize=(18,12))
plt.subplot(1,3,1)
plt.title("Result from custom function")
plt.imshow(gray,cmap="gray")
plt.subplot(1,3,2)
plt.title("Result from OpenCV function")
plt.imshow(gray_cv,cmap="gray")
plt.subplot(1,3,3)
plt.title("Difference")
plt.imshow(np.abs(gray-gray_cv),cmap="gray")
plt.show()
