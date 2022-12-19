import numpy as np
import cv2
import matplotlib.pyplot as plt

def normalize_points(pts, numDims): 
    # strip off the homogeneous coordinate
    points = pts[:numDims,:]

    # compute centroid
    cent = np.mean(points, axis=1)

    # translate points so that the centroid is at [0,0]
    translatedPoints = np.transpose(points.T - cent)

    # compute the scale to make mean distance from centroid sqrt(2)
    meanDistanceFromCenter = np.mean(np.sqrt(np.sum(np.power(translatedPoints,2), axis=0)))
    if meanDistanceFromCenter > 0: # protect against division by 0
        scale = np.sqrt(numDims) / meanDistanceFromCenter
    else:
        scale = 1.0

    # compute the matrix to scale and translate the points
    # the matrix is of the size numDims+1-by-numDims+1 of the form
    # [scale   0     ... -scale*center(1)]
    # [  0   scale   ... -scale*center(2)]
    #           ...
    # [  0     0     ...       1         ]    
    T = np.diag(np.array([*np.ones(numDims) * scale, 1], dtype=np.float))
    T[0:-1, -1] = -scale * cent

    if pts.shape[0] > numDims:
        normPoints = T @ pts
    else:
        normPoints = translatedPoints * scale

    # the following must be true:
    # np.mean(np.sqrt(np.sum(np.power(normPoints[0:2,:],2), axis=0))) == np.sqrt(2)

    return normPoints, T



def get_anaglyph(img1, img2):
    assert img1.shape[2] == 3
    assert img2.shape[2] == 3
    img1_ = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_ = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return np.dstack((img1_, img2_, img2_))



def draw_stereo_rectified_img(img1_rectified, img2_rectified):
    fig, ax = plt.subplots(1,2, figsize=(20,8))
    ax[0].imshow(img1_rectified)
    ax[0].set_title('Left image rectified')
    ax[1].imshow(img2_rectified)
    ax[1].set_title('Right image rectified')
    fig.tight_layout()
    return fig



def draw_disparity_map(disparity_map):
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    img = ax.imshow(disparity_map, cmap='turbo')
    ax.set_title('Disparity map')
    fig.colorbar(img, fraction=0.027, pad=0.01)
    fig.tight_layout()
    return fig



def compute_epe(disparity, gt_disparity):
    # compute end point error between disparity map and gt
    _epe = np.sqrt((disparity - gt_disparity) ** 2)
    epe = _epe.mean()
    epe3 = (_epe > 3.0).astype(np.float32).mean()
    
    return epe, epe3



def evaluate_criteria(extime, epe, epe3):
    if extime < 75.0:
        s = 'PASS'
        
        if epe < 3.0: score_epe = 15
        elif (epe >= 3.0) and (epe < 4.0): score_epe = 10
        elif (epe >= 4.0) and (epe < 5.0): score_epe = 5
        else: score_epe = 0

        if epe3 < 0.25: score_bad_pix = 15
        elif (epe3 >= 0.25) and (epe3 < 0.3): score_bad_pix = 10
        elif (epe3 >= 0.3) and (epe3 < 0.4): score_bad_pix = 5
        else: score_bad_pix = 0
    else:
        s = 'FAIL'
        score_epe = 0
        score_bad_pix = 0
    return s, score_epe, score_bad_pix