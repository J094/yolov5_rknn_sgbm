# J094
"""
Prepare datasets
"""

import time

import cv2

from utils.general import set_logging

LOGGER = set_logging(__name__)


class loadCam(object):
    def __init__(self, pipe='0', fps=5, imgProcessor=None):
        self.img_processor = imgProcessor

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        self.pipe = pipe

        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # set buffer size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS,fps)

    def __iter__(self):
        return self

    def __next__(self):
        dt = [0.0, 0.0]

        t0 = time.time()
        # read frame
        ret, img = self.cap.read()
        ret, img = self.cap.read()
        assert ret, 'Camera Error %d'%self.pipe
        t1 = time.time()
        dt[0] = t1 - t0
        
        imgL, imgR = self.img_processor.preprocess(img)
        dt[1] = time.time() - t1

        LOGGER.info(f"videocap: {dt[0]}, preprocess: {dt[1]}")
        return imgL, imgR

    def __len__(self):
        return 1
