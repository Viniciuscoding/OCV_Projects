import numpy as np
import matplotlib.pyplot as plt
import cv2

import matplotlib
matplotlib.rcParams['figure.figsize'] = (18.0, 12.0)
matplotlib.rcParams['image.interpolation'] = 'bilinear'

# Supress warnings
import warnings
warnings.filterwarnings("ignore")

# Read the image. Remember OpenCV reads it in BGR.
img = cv2.imread("images/Coins.png")

plt.imshow(img[:,:,::-1])
plt.show()

 
# BGR to HSV formula

# V←max(R,G,B)

# S←⎧ V − min(R,G,B)
#   ⎩ V0 if V≠0 otherwise

# H←⎧if V == 0 then H = 0
#   ⎪60(G−B)/(V−min(R,G,B))     if V=R
#   ⎨120+60(B−R)/(V−min(R,G,B)) if V=G
#   ⎪240+60(R−G)/(V−min(R,G,B)) if V=B
#   ⎩If H<0 then H = H + 360 on output 0≤V≤1, 0≤S≤1, 0≤H≤360 


# The values are converted back to 0 and 255 and uint8 for image display:

# 8-bit images: V←255V, S←255S, H←H/2(to fit to 0 to 255)
# 16-bit images: (currently not supported) V←65535V, S←65535S, H←H
# 32-bit images: H, S, and V are left as is


def convertBGRtoHSV(image):
    # Split the channels
    imgB, imgG, imgR = cv2.split(image)
    # Normalize them to between 0 and 1 as a float number
    b, g, r = imgB/255.0, imgG/255.0, imgR/255.0
    
    # Get the height(h) and width(w) of the image
    h, w, _ = image.shape
    
    # Create an empty image of the same size of the original image
    hsv = np.zeros((h, w, 3), dtype = np.uint8)
    
    H, S, V = 0.0, 0.0, 0.0
    for col in range(h):
        for row in range(w):
            R, G, B = r[col,row], g[col,row], b[col,row]

            Vmin = min(R, G, B)
            V = Vmax = max(R, G, B)

            if Vmax != 0:
                S = (Vmax - Vmin) / Vmax
            else:
                S = 0

            if H < 0:
                H = H + 360
            elif Vmax == Vmin:
                H = 0
            elif Vmax == R:
                H = 60*(G-B)/(Vmax-Vmin)
            elif Vmax == G:
                H = 120 + 60*(B-R)/(Vmax-Vmin)
            elif Vmax == B:
                H = 240 + 60*(R-G)/(Vmax-Vmin)
            
            hsv[col, row, 0] = np.round(H/2).astype("uint8")
            hsv[col, row, 1] = np.round(S*255).astype("uint8")
            hsv[col, row, 2] = np.round(V*255).astype("uint8")

    return hsv


hsv = convertBGRtoHSV(img)

hsv_cv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# Displaying the results
plt.subplot(1,3,1)
plt.title("Result from custom function")
plt.imshow(hsv[:,:,::-1])
plt.subplot(1,3,2)
plt.title("Result from OpenCV function")
plt.imshow(hsv_cv[:,:,::-1])
plt.subplot(1,3,3)
plt.title("Difference")
plt.imshow(np.abs(hsv-hsv_cv)[:,:,::-1])
plt.show()


# HSI Formula   

#     #Hue
#     numi = 1/2*((R-G)+(R-B));
#     denom = ((R-G).^2+((R-B).*(G-B))).^0.5;

#     #To avoid divide by zero exception add a small number in the denominator
#     H = acosd(numi./(denom+0.000001));

#     #If B>G then H= 360-Theta
#     H(B>G) = 360-H(B>G)

#     #Normalize to the range [0 1]
#     H = H/360;

#     #Saturation
#     S=1 - (3./(sum(I,3)+0.000001)).*min(I,[],3);

#     #Intensity
#     I = sum(I,3)./3;

#     #HSI
#     HSI=zeros(size(A));
#     HSI(:,:,1)=H;
#     HSI(:,:,2)=S;
#     HSI(:,:,3)=I;
