from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Path to image scanned")
args = vars(ap.parse_args())
# print(args)

image = cv2.imread(args['image'])
# image is a 3d matrix of BGR value of each Pixel
ratio = image.shape[0] / 500.0
# image.shape returns height, width, colors = 3
# print('image.shape: ', image.shape)
orig = image.copy()
# using adrian's image resize function
image = imutils.resize(image, height=500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# perform 5x5 gaussian blur
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# perform Canny Edge Detection
edged = cv2.Canny(gray, 75, 200)

# Print Step 1 Edge Detection
print('STEP 1 Edge Detection')
cv2.imshow('Image-Original', image)
cv2.imshow('Edged', edged)

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx) == 4:
        screenCnt = approx
        # print(screenCnt)
        break

print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow('Countours', image)

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped > T).astype("uint8") * 255

print("STEP 3: Perspective transform")
cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))

cv2.waitKey(0)
cv2.destroyAllWindows()
