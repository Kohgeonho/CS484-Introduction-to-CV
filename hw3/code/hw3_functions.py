################################################################
# WARNING
# --------------------------------------------------------------
# When you submit your code, do NOT include blocking functions
# in this file, such as visualization functions (e.g., plt.show, cv2.imshow).
# You can use such visualization functions when you are working,
# but make sure it is commented or removed in your final submission.
#
# Before final submission, you can check your result by
# set "VISUALIZE = True" in "hw3_main.py" to check your results.
################################################################
from tkinter import W
from tkinter.tix import WINDOW
from cv2 import normalize
from utils import normalize_points, draw_disparity_map
import numpy as np
import cv2
import matplotlib.pyplot as plt

#=======================================================================================
# Your best hyperparameter findings here
WINDOW_SIZE = 41
DISPARITY_RANGE = 40
AGG_FILTER_SIZE = 31



#=======================================================================================
def bayer_to_rgb_bilinear(bayer_img):
    ################################################################

    img_col, img_row = bayer_img.shape
    tile_size = [img_col // 2, img_row // 2]
    r_mask = np.tile(np.array([[1,0],[0,0]]), tile_size)
    g_mask = np.tile(np.array([[0,1],[1,0]]), tile_size)
    b_mask = np.tile(np.array([[0,0],[0,1]]), tile_size)

    r_img = bayer_img * r_mask
    g_img = bayer_img * g_mask
    b_img = bayer_img * b_mask

    rb_conv = np.array([
        [0.25, 0.5,  0.25],
        [0.5,  1.,   0.5 ],
        [0.25, 0.5,  0.25],
    ])
    g_conv = np.array([
        [0.,   0.25, 0.  ],
        [0.25, 1.,   0.25],
        [0.,   0.25, 0.  ],
    ]) ## need to handle edge cases (3-mean cases)

    def conv(img, filter):
        padded_img = np.pad(img, 1, "reflect")
        return np.array([
            padded_img[i:i+img_col, j:j+img_row] * filter[i, j]
            for i in range(3)
            for j in range(3)
        ]).sum(axis=0).astype(np.uint8)

    rgb_img = np.array([
        conv(r_img, rb_conv),
        conv(g_img, g_conv),
        conv(b_img, rb_conv),
    ]).transpose((1,2,0))

    ################################################################
    return rgb_img



#=======================================================================================
def bayer_to_rgb_bicubic(bayer_img):
    # Your code here
    ################################################################
    rgb_img = None


    ################################################################
    return rgb_img



#=======================================================================================
def calculate_fundamental_matrix(pts1, pts2):
    # Assume input matching feature points have 2D coordinate
    assert pts1.shape[1]==2 and pts2.shape[1]==2
    # Number of matching feature points should be same
    assert pts1.shape[0]==pts2.shape[0]
    # Your code here
    ################################################
    n = pts1.shape[0]
    pts1_3d = np.hstack((pts1, np.ones((n,1))))
    pts2_3d = np.hstack((pts2, np.ones((n,1))))
    pts1_norm, T1 = normalize_points(pts1_3d.T, 2)
    pts2_norm, T2 = normalize_points(pts2_3d.T, 2)

    A = np.array([
        (x1.reshape(3,1) @ x2.reshape(1,3)).reshape(-1)
        for x1, x2 in zip(pts1_norm.T, pts2_norm.T)
    ])

    w, v = np.linalg.eig(A.T @ A)
    F = v[:, np.argmin(w)].reshape(3,3)

    U, S, Vh = np.linalg.svd(F)
    S[np.argmin(S)] = 0
    F = U @ np.diag(S) @ Vh
    
    fundamental_matrix = T2.T @ F.T @ T1

    ################################################################
    return fundamental_matrix



#=======================================================================================
def rectify_stereo_images(img1, img2, h1, h2):
    # Your code here
    # You should get un-cropped image.
    # In order to superpose two rectified images, you need to create certain amount of margin.
    # Which means you need to do some additional things to get fully warped image (not cropped).
    ################################################

    img1_h, img1_w, _ = img1.shape
    img2_h, img2_w, _ = img2.shape

    corner1 = np.array([[0, 0], [0, img1_h], [img1_w, 0], [img1_w, img1_h]], dtype=np.float32).reshape(-1,1,2)
    corner2 = np.array([[0, 0], [0, img2_h], [img1_w, 0], [img2_w, img2_h]], dtype=np.float32).reshape(-1,1,2)

    img1_corner = cv2.perspectiveTransform(corner1, h1)
    img2_corner = cv2.perspectiveTransform(corner2, h2)
    
    img_lu_corner = np.concatenate([img1_corner, img2_corner], axis=0).min(axis=0)
    img_rb_corner = np.concatenate([img1_corner, img2_corner], axis=0).max(axis=0)
    rectified_h, rectified_w = np.ceil(img_rb_corner - img_lu_corner).astype(int)[0]

    left_edge, right_edge = img_lu_corner[0]
    move_m = np.float32([
        [1, 0, -left_edge + 10],
        [0, 1, -right_edge + 10], 
        [0, 0, 1],
    ])
    img1_rectified = cv2.warpPerspective(img1, h1 @ move_m, (rectified_h + 20, rectified_w + 20))
    img2_rectified = cv2.warpPerspective(img2, h2 @ move_m, (rectified_h + 20, rectified_w + 20))

    ################################################
    return img1_rectified, img2_rectified




#=======================================================================================
def calculate_disparity_map(img1, img2):
    # First convert color image to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # You have to get disparity (depth) of img1 (left)
    # i.e., I1(u) = I2(u + d(u)),
    # where u is pixel positions (x,y) in each images and d is dispairty map.
    # Your code here
    ################################################
    w = WINDOW_SIZE
    a = AGG_FILTER_SIZE
    
    avg_filter = -np.ones((w,w)) / (w**2)
    avg_filter[w//2, w//2] += 1

    img1_diff = cv2.filter2D(img1_gray, -1, avg_filter).astype(np.int32)
    img2_diff = cv2.filter2D(img2_gray, -1, avg_filter).astype(np.int32)

    norm1 = np.sqrt(cv2.filter2D(img1_diff ** 2, -1, np.ones((w, w)) / (w * w))).astype(np.float)
    norm2 = np.sqrt(cv2.filter2D(img2_diff ** 2, -1, np.ones((w, w)) / (w * w))).astype(np.float)

    def NCC(img1_, img2_, d):
        img_mul = np.roll(img1_, -d) * img2_
        img_mul_norm = cv2.filter2D(img_mul, -1, np.ones((w,w)) / (w * w))
        return img_mul_norm / (np.roll(norm1, -d) * norm2 + 1e-5)

    disparity_map = -np.array([
        cv2.filter2D(NCC(img1_diff, img2_diff, d).astype(np.float), -1, np.ones((a, a)) / (a * a))
        for d in range(1, DISPARITY_RANGE)
    ]).argmax(axis=0)

    ################################################################
    return disparity_map


#=======================================================================================
# Anything else:
