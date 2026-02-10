import cv2
import numpy as np


def find_homography(img1, img2, points):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect features and compute descriptors using ORB
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match features using Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # NOTE: interestingly, sorting by distance (how close the descriptors are) seemingly worsened the results consistently,
    # especially in the points near the left window.
    # seems like it shouldn't matter since RANSAC should sample randomly?
    # maybe just a coincidence 
    # matches = sorted(matches, key=lambda x: x.distance)

    # Visualize matches 
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("matches", match_img)
    cv2.waitKey(0)
    cv2.destroyWindow("matches")

    # Extract coords of matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the homography matrix using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H


points = []
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(param, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow('select points (press any key to finish)', param)


def main():
    img1 = cv2.imread('view_1.png')
    img2 = cv2.imread('view_2.png')

    # Select points to transform
    img1_copy = img1.copy()
    cv2.imshow('select points (press any key to finish)', img1_copy)
    cv2.setMouseCallback('select points (press any key to finish)', select_point, img1_copy)

    cv2.waitKey(0)
    cv2.destroyWindow('select points (press any key to finish)')

    # Transform points from img1 to img2
    H = find_homography(img1, img2, points)
    transformed_points = cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), H)
    
    # Visualize original and transformed points
    img1_copy = img1.copy()
    for pt in points:
        cv2.circle(img1_copy, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
    for pt in transformed_points.reshape(-1, 2):
        cv2.circle(img2, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), -1)

    cv2.imshow('original points', img1_copy)
    cv2.imshow('transformed points', img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Visualize warped image vs actual image
    warped_img = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

    cv2.imshow('warped image', warped_img)
    cv2.waitKey(0)
    cv2.destroyWindow('warped image')


if __name__ == "__main__":
    main()