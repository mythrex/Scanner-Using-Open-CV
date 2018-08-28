from pyimagesearch.transform import four_point_transform
import numpy as np
import argparse
import cv2
import imutils

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Path to image scanned")
args = vars(ap.parse_args())
print(args)
