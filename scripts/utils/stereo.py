# J094
"""
Perform stereo vision algorithms
"""

from pathlib import Path
import json

import numpy as np
import cv2

from utils.general import set_logging

LOGGER = set_logging(__name__)


def obtain_depth(coord3D, objBox, ratio=0.8):
    # basic shift of reference boxes
    dx, dy = int((objBox[2]-objBox[0])/3), int((objBox[3]-objBox[1])/3)
    rdx, rdy = int(ratio*dx/2), int(ratio*dy/2)
    xc, yc = int((objBox[2] + objBox[0])/2), int((objBox[3] + objBox[1])/2)
    # 9 centers from object box
    # 0: center, 1: left top, 2: left bottom, 3: right bottom, 4: right top
    refCenters = [
        (xc-dx, yc-dy), (xc   , yc-dy), (xc+dx, yc-dy),
        (xc-dx, yc   ), (xc   , yc   ), (xc+dx, yc   ),
        (xc-dx, yc+dy), (xc   , yc+dy), (xc+dx, yc+dy),
                ]
    # obtain reference boxes
    refBoxes = []
    for c in refCenters:
        refBox = (c[0]-rdx, c[1]-rdy, c[0]+rdx, c[1]+rdy)
        refBoxes.append(refBox)
    # print(dx, ' ', dy)
    # print(rdx, ' ', rdy)
    # print(refBoxes)
    # calculate median for each refBox
    # if median of a certain refBox is >= 200 (2 meter), then drop this result as outliers
    d_m = []
    for b in refBoxes:
        refDepth = coord3D[b[1]:b[3]+1, b[0]:b[2], 2]
        temp_m = np.median(refDepth)
        if temp_m < 200: d_m.append(temp_m)
    # return the min of all available refBox
    return int(np.amin(d_m)*1.4)/100 if d_m else 1000


class imageProcessor(object):
    def __init__(self, readPath: str, imgSize=(416,312)) -> None:
        self.readPath = Path(readPath)
        self.cameraDict = self._readJSON()
        self.rectifyMaps = dict()
        # calculate and store RectifyMaps once
        self._getRectifyMaps()
        self.imgSize = imgSize

    def _readJSON(self):
        readFile = self.readPath / 'stereo_parameters_1280x480_new.json'
        with open(readFile, 'r') as rfile:
            cameraDict = json.load(rfile)
        # transform list to np.array
        for key in cameraDict:
            if isinstance(cameraDict[key], list):
                cameraDict[key] = np.asarray(cameraDict[key])
        return cameraDict
    
    def _getRectifyMaps(self):
        mapLx, mapLy = cv2.initUndistortRectifyMap(
            self.cameraDict['mtxL'], self.cameraDict['distL'], 
            self.cameraDict['R1'], self.cameraDict['P1'], 
            tuple(self.cameraDict['imgSize']), cv2.CV_32FC1,
            )
        mapRx, mapRy = cv2.initUndistortRectifyMap(
            self.cameraDict['mtxR'], self.cameraDict['distR'], 
            self.cameraDict['R2'], self.cameraDict['P2'], 
            tuple(self.cameraDict['imgSize']), cv2.CV_32FC1,
            )
        self.rectifyMaps['mapLx'] = mapLx
        self.rectifyMaps['mapRx'] = mapRx
        self.rectifyMaps['mapLy'] = mapLy
        self.rectifyMaps['mapRy'] = mapRy

    def preprocess(self, img):
        w, h = self.cameraDict['imgSize']
        # cut image to left and right parts
        imgL = img[:, :w]
        imgR = img[:, w:]
        # undistort and rectify images
        imgLRe = cv2.remap(
            imgL, self.rectifyMaps['mapLx'], self.rectifyMaps['mapLy'],
            interpolation=cv2.INTER_AREA,
            )
        imgRRe = cv2.remap(
            imgR, self.rectifyMaps['mapRx'], self.rectifyMaps['mapRy'],
            interpolation=cv2.INTER_AREA,
            )
        # resize image
        if w != self.imgSize[0] or h != self.imgSize[1]:
            imgLRe = cv2.resize(imgLRe, self.imgSize, interpolation=cv2.INTER_AREA)
            imgRRe = cv2.resize(imgRRe, self.imgSize, interpolation=cv2.INTER_AREA)
        return imgLRe, imgRRe
   
    # calculate desparity and depth map
    def stereoMatchSGBM(self, left_image, right_image, down_scale=False):
        # config parameters
        if left_image.ndim == 2:
            img_channels = 1
        else:
            img_channels = 3
        min_disp = 1
        num_disp = 48 - min_disp
        blockSize = 9
        paraml = {
            'minDisparity': min_disp,
            'numDisparities': num_disp,
            'blockSize': blockSize,
            'P1': 8 * img_channels * blockSize ** 2,
            'P2': 32 * img_channels * blockSize ** 2,
            'disp12MaxDiff': 3,
            'preFilterCap': 63,
            'uniquenessRatio': 5,
            'speckleWindowSize': 100,
            'speckleRange': 3,
            'mode': cv2.StereoSGBM_MODE_SGBM_3WAY,
            }
        left_matcher = cv2.StereoSGBM_create(**paraml)
        size = (left_image.shape[1], left_image.shape[0])
        # calculate disparity
        if down_scale == False:
            disparity_left = left_matcher.compute(left_image, right_image)
        else:
            left_image_down = cv2.pyrDown(left_image)
            right_image_down = cv2.pyrDown(right_image)
            factor = left_image.shape[1] / left_image_down.shape[1]
            disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
            disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
            disparity_left = factor * disparity_left
        # calculate real disparity
        disp_left = disparity_left.astype(np.float32) / 16.0
        coord_3d = cv2.reprojectImageTo3D(disp_left, self.cameraDict['Q'], handleMissingValues=True)
        return disp_left, coord_3d

