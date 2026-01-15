# demo.py
# This script demonstrates how to compute a homography matrix
# to convert between factory ground coordinates and pixel coordinates from security camera images


import cv2
import numpy as np
import json
import os

# Load points from points.json
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, 'points.json')

with open(json_path, 'r') as f:
    points_data = json.load(f)

model_points_list = []
image_points_list = []

for name, data in points_data.items():
    world_coords = data['world']
    pixel_coords = data['pixels']
    
    # Check if point is on the ground (Z-coordinate close to 0)
    # Using 0.1 as threshold for floating point comparisons
    if abs(world_coords[2]) < 0.1:
        model_points_list.append(world_coords[:2]) # Taking X and Y
        image_points_list.append(pixel_coords)

model_points = np.array(model_points_list, dtype=float)
image_points = np.array(image_points_list, dtype=float)

# Refine points using cornerSubPix
def refine_points(image_points, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined_pts = cv2.cornerSubPix(gray, image_points.astype(np.float32), (5, 5), (-1, -1), term)
    return refined_pts

# # Load image for refinement
# img = cv2.imread("frame.png")
# image_points = refine_points(image_points, img)

# Calculate the Homography Matrix
# RANSAC helps exclude outliers, not helpful with 4 points (minimum)
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


# Visualize the points on the image
img = cv2.imread("frame.png")
for pt in image_points:
    cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)


# Show test point
u, v = int(pixel_coord[0][0][0]), int(pixel_coord[0][0][1])
cv2.circle(img, (u, v), 7, (0, 255, 0), -1)
cv2.putText(img, "Test Point", (u+10, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


# Save to file
cv2.imwrite("homography_demo_output.png", img)
