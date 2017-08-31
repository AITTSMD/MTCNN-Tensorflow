# coding: utf-8
"""
    functions
"""

import os
import cv2
import numpy as np


def show_landmark(face, landmark):
    """
        view face with landmark for visualization
    """
    face_copied = face.copy().astype(np.uint8)
    for (x, y) in landmark:
        xx = int(face.shape[0]*x)
        yy = int(face.shape[1]*y)
        cv2.circle(face_copied, (xx, yy), 2, (0,0,0), -1)
    cv2.imshow("face_rot", face_copied)
    cv2.waitKey(0)


#rotate(img, f_bbox,bbox.reprojectLandmark(landmarkGt), 5)
#img: the whole image
#BBox:object
#landmark:
#alpha:angle
def rotate(img, bbox, landmark, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
    """
    center = ((bbox.left+bbox.right)/2, (bbox.top+bbox.bottom)/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    #whole image rotate
    #pay attention: 3rd param(col*row)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat,(img.shape[1],img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
    #crop face 
    face = img_rotated_by_alpha[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
    return (face, landmark_)


def flip(face, landmark):
    """
        flip face
    """
    face_flipped_by_x = cv2.flip(face, 1)
    #mirror
    landmark_ = np.asarray([(1-x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
    landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth
    return (face_flipped_by_x, landmark_)

def randomShift(landmarkGt, shift):
    """
        Random Shift one time
    """
    diff = np.random.rand(5, 2)
    diff = (2*diff - 1) * shift
    landmarkP = landmarkGt + diff
    return landmarkP

def randomShiftWithArgument(landmarkGt, shift):
    """
        Random Shift more
    """
    N = 2
    landmarkPs = np.zeros((N, 5, 2))
    for i in range(N):
        landmarkPs[i] = randomShift(landmarkGt, shift)
    return landmarkPs
