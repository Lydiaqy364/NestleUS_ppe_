# demo.py
# This script demonstrates how to compute a homography matrix
# to convert between factory ground coordinates and pixel coordinates from security camera images


import cv2
import numpy as np


# Coordinates (x,y) from CloudCompare
model_points = np.array([
    [31.08, 8.64],  # bottom right door corner
    [30.69, 9.59],  # bottom right stair corner
    [32.57, 9.16],  # bottom right yellow column
    [32.07, 8.63],  # bottom left door corner
], dtype=float)


# Corresponding coordinates from image (in pixels)
image_points = np.array([
    [412, 1208],
    [542, 1230],
    [267, 1374],
    [288, 1285]
], dtype=float)


# Calculate the Homography Matrix
H, mask = cv2.findHomography(model_points, image_points, cv2.RANSAC)
print("Homography Matrix:\n", H)


# Factory point to pixel conversion example
test_point = np.array([[[31.896, 13.875]]], dtype=float)
pixel_coord = cv2.perspectiveTransform(test_point, H)
print(f"Factory point (31.896, 13.875) is at Pixel: {pixel_coord}")


# Pixel to factory point conversion example
test_pixel = np.array([[[1060, 1730]]], dtype=float)
factory_coord = cv2.perspectiveTransform(test_pixel, np.linalg.inv(H))
print(f"Pixel point (1060, 1730) is at Factory Coord: {factory_coord}")