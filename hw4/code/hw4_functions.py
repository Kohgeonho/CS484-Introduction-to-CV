import cv2
import numpy as np

ALPHA = 0.03
KEYPOINT_THRESHOLD = 0.005
LOCAL_WINDOW_RATIO = 0.25
GAUSSIAN_SIGMA = 0.4
CLAMP_THRESHOLD = 0.2
NNDR_THRESHOLD = 0.85

def get_interest_points(image, descriptor_window_image_width):
    # Local Feature Stencil Code
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of interest points for the input image

    # 'image' can be grayscale or color, your choice.
    # 'descriptor_window_image_width', in pixels.
    #   This is the local feature descriptor width. It might be useful in this function to
    #   (a) suppress boundary interest points (where a feature wouldn't fit entirely in the image, anyway), or
    #   (b) scale the image filters being used.
    # Or you can ignore it.

    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.

    # Implement the Harris corner detector (See Szeliski 4.1.1) to start with.

    # If you're finding spurious interest point detections near the boundaries,
    # it is safe to simply suppress the gradients / corners near the edges of
    # the image.

    # Placeholder that you can delete -- random points

    ## Cut the width and height of the image to be a multiple of the window size.
    wsize = descriptor_window_image_width
    
    h, w = image.shape
    h = (h // wsize) * wsize
    w = (w // wsize) * wsize
    image = image[:h, :w]

    ## Find the candidate keypoints.
    x_gradient = cv2.filter2D(image.T, -1, np.array([-2, -1, 0, 1, 2])).T
    y_gradient = cv2.filter2D(image, -1, np.array([-2, -1, 0, 1, 2]))

    unit_matrix_A = np.array([
        x_gradient ** 2,
        x_gradient * y_gradient,
        y_gradient ** 2,    
    ]).transpose(1,2,0)
    matrix_A = cv2.GaussianBlur(unit_matrix_A, (wsize-1, wsize-1), 1)

    determinant_A = matrix_A[:,:,0] * matrix_A[:,:,2] - matrix_A[:,:,1] * matrix_A[:,:,1]
    trace_A = (matrix_A[:,:,0] + matrix_A[:,:,2])

    matrix_R = determinant_A - ALPHA * (trace_A ** 2)
    keypoint_mask = matrix_R > KEYPOINT_THRESHOLD

    ## Find Local maximum keypoints.
    keypoint_matrix = matrix_R * keypoint_mask
    wsize = int(wsize * LOCAL_WINDOW_RATIO)
    h_block, w_block = h//wsize, w//wsize
    local_max_idx = keypoint_matrix.reshape((h_block, wsize, w_block, wsize)) \
                                   .transpose((0,2,1,3)) \
                                   .reshape(h_block, w_block, wsize**2) \
                                   .argmax(axis=2)
    local_max_mask = np.zeros((h_block, w_block, wsize**2))
    local_max_mask[
        np.repeat(np.arange(h_block), w_block),
        np.tile(np.arange(w_block), h_block),
        local_max_idx.reshape(-1)
    ] = 1
    local_max_mask = local_max_mask.reshape(h_block, w_block, wsize, wsize) \
                                   .transpose((0,2,1,3)) \
                                   .reshape(h_block * wsize, w_block * wsize)
    keypoint_mask = keypoint_mask * local_max_mask
    cv2.imwrite("image2_local_max_keypoint_mask.png", keypoint_mask * 255)

    y, x = keypoint_mask.nonzero()

    return x,y

    # After computing interest points, here's roughly how many we return
    # For each of the three image pairs
    # - Notre Dame: ~1300 and ~1700
    # - Mount Rushmore: ~3500 and ~4500
    # - Episcopal Gaudi: ~1000 and ~9000


def get_descriptors(image, x, y, descriptor_window_image_width):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of feature descriptors for a given set of interest points.

    # 'image' can be grayscale or color, your choice.
    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
    #   The local features should be centered at x and y.
    # 'descriptor_window_image_width', in pixels, is the local feature descriptor width.
    #   You can assume that descriptor_window_image_width will be a multiple of 4
    #   (i.e., every cell of your local SIFT-like feature will have an integer width and height).
    # If you want to detect and describe features at multiple scales or
    # particular orientations, then you can add input arguments.

    # 'features' is the array of computed features. It should have the
    #   following size: [length(x) x feature dimensionality] (e.g. 128 for
    #   standard SIFT)

    h, w = image.shape

    # delete x's and y's in the edge of the image.
    wsize = descriptor_window_image_width
    radius = descriptor_window_image_width // 2

    x = x.astype(int)
    y = y.astype(int)

    x[x < radius] = radius
    x[x > w - radius] = w - radius
    y[y < radius] = radius
    y[y > h - radius] = h - radius
    
    x_gradient = cv2.filter2D(image.T, -1, np.array([-2, -1, 0, 1, 2])).T
    y_gradient = cv2.filter2D(image, -1, np.array([-2, -1, 0, 1, 2]))

    m = np.linalg.norm(np.array([x_gradient, y_gradient]), axis=0)
    theta = np.arctan2(x_gradient, y_gradient)

    def keypoint_descriptor(r):
        xi, yi = r

        m_window = m[yi-radius:yi+radius, xi-radius:xi+radius]
        theta_window = theta[yi-radius:yi+radius, xi-radius:xi+radius]

        m_window = cv2.GaussianBlur(m_window, (wsize-1, wsize-1), GAUSSIAN_SIGMA)
        theta_window = cv2.GaussianBlur(theta_window, (wsize-1, wsize-1), GAUSSIAN_SIGMA)

        m_window = m_window.reshape(4, wsize//4, 4, wsize//4).transpose(0,2,1,3).reshape(16, wsize**2//16)
        theta_window = theta_window.reshape(4, wsize//4, 4, wsize//4).transpose(0,2,1,3).reshape(16, wsize**2//16)

        features = np.zeros((16,8))
        for i in range(16):
            features[i] = np.histogram(theta_window[i], bins=8, range=(-np.pi, np.pi), weights=m_window[i])[0]
        
        return features.reshape(-1)

    features = np.apply_along_axis(
        keypoint_descriptor,
        axis=0,
        arr=np.array([x,y]),
    ).T

    return features


def match_features(features1, features2):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Please implement the "nearest neighbor distance ratio test",
    # Equation 4.18 in Section 4.1.3 of Szeliski.

    #
    # Please assign a confidence, else the evaluation function will not work.
    #

    # This function does not need to be symmetric (e.g., it can produce
    # different numbers of matches depending on the order of the arguments).

    # Input:
    # 'features1' and 'features2' are the n x feature dimensionality matrices.
    #
    # Output:
    # 'matches' is a k x 2 matrix, where k is the number of matches. The first
    #   column is an index in features1, the second column is an index in features2.
    #
    # 'confidences' is a k x 1 matrix with a real valued confidence for every match.

    # Placeholder random matches and confidences.

    # feature dimensionality : 128
    num_features = features1.shape[0]
    print(features1.shape[0], features2.shape[0])

    ## Cosine Similarity & Clampping
    features1 = features1 / (np.linalg.norm(features1, axis=1)[:, None] + 1e-5)
    features2 = features2 / (np.linalg.norm(features2, axis=1)[:, None] + 1e-5)
    features1[features1 >= CLAMP_THRESHOLD] = CLAMP_THRESHOLD
    features2[features2 >= CLAMP_THRESHOLD] = CLAMP_THRESHOLD
    features1 = features1 / (np.linalg.norm(features1, axis=1)[:, None] + 1e-5)
    features2 = features2 / (np.linalg.norm(features2, axis=1)[:, None] + 1e-5)

    distances = features1 @ features2.T

    nearest_one_idx = np.argmax(distances, axis=1)
    nearest_two = np.partition(distances, -2, axis=1)[:, -2:]
    nearest_two = np.sort(nearest_two, axis=1)
    nndr = nearest_two[:, 0] / (nearest_two[:, 1] + 1e-5)

    match_idx = (nndr < NNDR_THRESHOLD).nonzero()[0]

    matches = np.array([
        match_idx,
        nearest_one_idx[match_idx],
    ]).T
    confidences = 1 - nndr[match_idx]

    return matches, confidences
