#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import cv2
import argparse
import sys

# from calibration_store import load_stereo_coefficients

def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!


    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg



# In[7]:


#cap1 = cv2.VideoCapture(0)
#cap2 = cv2.VideoCapture(1)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
cap1 = cv2.VideoCapture('leftOut.avi')
cap2 = cv2.VideoCapture('rightOut.avi')
#fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
#fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
fourcc3 = cv2.VideoWriter_fourcc(*'XVID')
#out1 = cv2.VideoWriter('output1.avi',fourcc1, 20.0, (640,480))
#out2 = cv2.VideoWriter('output2.avi',fourcc2, 20.0, (640,480))
out3 = cv2.VideoWriter('disparity.avi',fourcc3, 20.0, (640,480))
focal=3.7
baseline = 70
while(True):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    dispa=depth_map(gray1, gray2)
    #depth = baseline * focal / dispa
    disparity_image = cv2.applyColorMap(dispa, cv2.COLORMAP_JET)

    #out1.write(frame1)
    #out2.write(frame2)
    out3.write(disparity_image)
    cv2.imshow('frame1',frame1)
    cv2.imshow('frame2',frame2)
    cv2.imshow('frame3',disparity_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()





# In[ ]:




