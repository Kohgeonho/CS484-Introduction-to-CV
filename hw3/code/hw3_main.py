################################################################
# CS484 HW3
################################################################
# WARINING
# --------------------------------------------------------------
# We will only use your "hw3_functions.py" to evaluate your assignment.
# Anything you change in this main file will not be used.
################################################################
import os
from pickletools import uint8
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import traceback
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from hw3_functions import (
    bayer_to_rgb_bilinear,
    bayer_to_rgb_bicubic,
    calculate_fundamental_matrix,
    rectify_stereo_images,
    calculate_disparity_map
)
from utils import *


################################################################
# Modify area
# --------------------------------------------------------------
# Use boolean value VISUALIZE to visualize your work.
# Even this value is set to False, results will be written in result folder.
VISUALIZE = False
# If you are working on docker / virtual machine / Jupyter Notebook,
# You may have to do some additional things here to show the plot.
# e.g.) %matplotlib inline, matplotlib.use('QtAgg')
# Note that TA will run this with VISUALIZE = FALSE
################################################################


class HW3Stereo:
    result_dir = '../result'
    os.makedirs(result_dir, exist_ok=True)

    #=======================================================================================
    # Read bayer pattern image
    data_dir = '../data'
    img1_bayer = cv2.imread(f'{data_dir}/img1_bayer.png', -1)
    img2_bayer = cv2.imread(f'{data_dir}/img2_bayer.png', -1)

    # Read feature point
    points = np.loadtxt(f'{data_dir}/feature_points.txt', dtype=np.float32, delimiter=',')
    pts1 = points[:,:2]
    pts2 = points[:,2:]

    # img3 (left) and 4 (right) are perfectly rectified image
    img3 = cv2.imread(f'{data_dir}/img3.png', -1)
    img4 = cv2.imread(f'{data_dir}/img4.png', -1)
    gt_disparity = cv2.imread(f'{data_dir}/img3_disp.exr', -1)


    def run(self):
        print('--------'*8)
        #=======================================================================================
        # HW3-a
        try:
            # Bilinear
            self.img1_bilinear = bayer_to_rgb_bilinear(self.img1_bayer)
            self.img2_bilinear = bayer_to_rgb_bilinear(self.img2_bayer)

            # Bicubuc is optional. We will use bilinear for following steps
            self.img1_bicubic = bayer_to_rgb_bicubic(self.img1_bayer)
            self.img2_bicubic = bayer_to_rgb_bicubic(self.img2_bayer)

            # Write / visualize images
            cv2.imwrite(f'{self.result_dir}/bilinear_img1.png', self.img1_bilinear[:,:,::-1])
            cv2.imwrite(f'{self.result_dir}/bilinear_img2.png', self.img2_bilinear[:,:,::-1])
            if (self.img1_bicubic is not None) and (self.img2_bicubic is not None):
                cv2.imwrite(f'{self.result_dir}/bicubic_img1.png', self.img1_bicubic[:,:,::-1])
                cv2.imwrite(f'{self.result_dir}/bicubic_img2.png', self.img2_bicubic[:,:,::-1])
        
        except: print(f'[ ERROR (a) ] {traceback.format_exc()}', end=f'{"--------"*8}\n')
        else: print('[ PASS (a) ]', end=f'\n{"--------"*8}\n')
        #=======================================================================================
        

        #=======================================================================================
        # HW3-b
        try:
            # Fundamental matrix
            self.fundamental_matrix = calculate_fundamental_matrix(self.pts1, self.pts2)
        
        except: print(f'[ ERROR (b) ] {traceback.format_exc()}', end=f'{"--------"*8}\n')
        else: print('[ PASS (b) ]', end=f'\n{"--------"*8}\n')
        #=======================================================================================


        #=======================================================================================
        # HW3-c
        # Compute homography matrix for rectification.
        # As described in website, computing this matrix requires much more steps
        # than what we covered in the lecture, you don't need to work on this part.
        try:
            _, h1, h2 = cv2.stereoRectifyUncalibrated(
                self.pts1, self.pts2,
                self.fundamental_matrix,
                (self.img1_bilinear.shape[1], self.img1_bilinear.shape[0]))

            self.img1_rectified, self.img2_rectified = rectify_stereo_images(self.img1_bilinear, self.img2_bilinear, h1, h2)

            # Write / visualize images
            fig = draw_stereo_rectified_img(self.img1_rectified, self.img2_rectified)
            fig.savefig(f'{self.result_dir}/rectified_imgs.png', bbox_inches='tight')
            cv2.imwrite(f'{self.result_dir}/rectified_img1.png', self.img1_rectified[:,:,::-1])
            cv2.imwrite(f'{self.result_dir}/rectified_img2.png', self.img2_rectified[:,:,::-1])
            if VISUALIZE: plt.show()
            anaglyph_rectified = get_anaglyph(self.img1_rectified, self.img2_rectified)
            cv2.imwrite(f'{self.result_dir}/rectified_anaglyph.png', anaglyph_rectified[:,:,::-1])

        except: print(f'[ ERROR (c) ] {traceback.format_exc()}', end=f'{"--------"*8}\n')
        else: print('[ PASS (c) ]', end=f'\n{"--------"*8}\n')
        #=======================================================================================
        

        #=======================================================================================
        # HW3-d
        # WARNING: This may take some time depending your implementation.
        # Your code must be done in "75 seconds". Otherwise you will not get score.
        try:
            tic = time.time()
            
            self.disparity_map = calculate_disparity_map(self.img3, self.img4)
            self.ex_time = time.time()-tic
            self.epe, self.epe3 = compute_epe(self.disparity_map, self.gt_disparity)

            eval_result = evaluate_criteria(self.ex_time, self.epe, self.epe3)
            print(f'[ {eval_result[0]} ] Disparity computation time: {self.ex_time:.2f} seconds')
            print(f'[ {eval_result[1]} pts ] EPE: {self.epe:.4f}')
            print(f'[ {eval_result[2]} pts ] Bad pixel ratio: {self.epe3*100:.2f}%')

            fig = draw_disparity_map(self.disparity_map)
            fig.savefig(f'{self.result_dir}/disparity_map.png', bbox_inches='tight')
            cv2.imwrite(f'{self.result_dir}/disparity_map.exr', self.disparity_map.astype(np.float32))
            if VISUALIZE: plt.show()
            
        except: print(f'[ ERROR (d) ] {traceback.format_exc()}', end=f'{"--------"*8}\n')
        else: print('[ PASS (d) ]', end=f'\n{"--------"*8}\n')
        #=======================================================================================






if __name__=='__main__':
    hw3 = HW3Stereo()
    hw3.run()


    