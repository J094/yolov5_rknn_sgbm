#!/usr/bin/env python
# J094
"""
Detector model for mower
"""

import os
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread, Condition

import cv2
import numpy as np
from rknnlite.api import RKNNLite
import rospy

from utils.general import set_logging, letterbox, draw, yolov5_post_process
from utils.stereo import imageProcessor, obtain_depth
from utils.dataset import loadCam
from utils.rosutils import rosPublisher, rosClient

LOGGER = set_logging(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

RKNN_MODEL = 'yolov5n_mower_640.rknn'


class detector(object):
    def __init__(
        self,
        source='0',
        fps=5,
        weightPath=ROOT / 'weights',
        paramPath=ROOT / 'params',
        imgSize=(640,480),
        saveImg=False,
        showImg=False,
        publish=False,
        client=False,
        ) -> None:
        self.source = source
        self.fps = fps
        self.weight_path = weightPath
        self.param_path = paramPath
        self.img_size = imgSize
        self.save_img = saveImg
        self.show_img = showImg
        self.publish = publish
        self.client = client

        # imgsz for letterbox
        sz = max(self.img_size)
        self.imgsz = (sz, sz)
 
        self.vid_writer = None
        self.pub = None

        if self.save_img:
            save_path = str(ROOT / 'demo.mp4')
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.vid_writer = cv2.VideoWriter(save_path, fourcc, 30, self.imgsz)

        if self.publish:
            self.pub = rosPublisher(self.fps)

        if self.client:
            self.c = rosClient()
 
        # load rknn model
        LOGGER.info("--> Loading RKNN model")
        rknn = RKNNLite()
        rknn.load_rknn(path=self.weight_path / RKNN_MODEL)
        # init runtime of rknn
        rknn.init_runtime()
        LOGGER.info("Load RKNN model done")

        # load image processor
        LOGGER.info("--> Loading image processor")
        img_processor = imageProcessor(readPath=self.param_path, imgSize=self.img_size)
        LOGGER.info("Load image processor done")

        self.rknn = rknn
        self.img_processor = img_processor

        LOGGER.info("--> Loading dataset")
        self.dataset = loadCam(self.source, self.fps, self.img_processor)
        LOGGER.info("Load dataset done")

        # time recorder
        self.dt, self.seen = [0.0, 0.0, 0.0, 0.0], 0

        self.queue4det = Queue(maxsize=1)

    def build_threads(self):
        self.thread4dataset = Thread(target=self.dataset_queue, daemon=True)
        self.thread4dataset.start()

    def dataset_queue(self):
       for imgL, imgR in self.dataset:
            # letterbox
            imgL, _, _ = letterbox(imgL, self.imgsz)
            imgR, _, _ = letterbox(imgR, self.imgsz)
            # if one queue is full, sleep to wait process
            while self.queue4det.full():
                time.sleep(0.0001)
            self.queue4det.put([imgL, imgR])
            time.sleep(0.0001)
 
    def det_process(self):
        self.build_threads()

        while True:
            self.seen += 1

            imgL, imgR = self.queue4det.get()

            t0 = time.time()
            # First-stage: yolov5 prediction
            # BGR2RGB
            im = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
            # prediction
            outputs = self.rknn.inference(inputs=[im])

            # if outputs is NoneType
            if not outputs: continue

            input_data = list()
            for input_data_i in outputs:
                input_data_i = input_data_i.reshape([3,-1]+list(input_data_i.shape[-2:]))
                input_data.append(np.transpose(input_data_i, (2, 3, 0, 1)))

            boxes, classes, scores = yolov5_post_process(input_data)
            t1 = time.time()
            self.dt[0] += t1 - t0

            # Second-stage: SGBM depth estimation
            # BGR2GRAY
            imL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            imR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            # estimation
            dispL, coord_3d = self.img_processor.stereoMatchSGBM(imL, imR, down_scale=True)
            t2 = time.time()
            self.dt[1] += t2 - t1

            # Third-stage: distance extraction
            dists = list()
            if boxes is not None:
                for box in boxes:
                    dist = obtain_depth(coord_3d, box, ratio=0.8)
                    # dists.append(dist)
                    # only publish dangerous distance
                    if dist <= 5000: dists.append(dist)
                if self.show_img and dists:
                    # draw boxes in img
                    draw(imgL, boxes, scores, classes, dists)
            t3 = time.time()
            self.dt[2] += t3 - t2
            self.dt[3] += t3 - t0

            # publish
            if self.publish:
                try:
                    if dists:
                        self.pub.pub_dist(np.amin(dists))
                    # test publish rate
                    else:
                        self.pub.pub_dist(100000)
                except rospy.ROSInterruptException:
                    pass
            LOGGER.info(f"yolov5: {self.dt[0]/self.seen}, sgbm: {self.dt[1]/self.seen}, \
                distance: {self.dt[2]/self.seen}, total: {self.dt[3]/self.seen}")

           # send req
            if self.client:
                if dists:
                    self.c.run(np.amin(dists))

            # show img
            if self.show_img:
                dispL = dispL.astype(np.uint8)
                heatmapL = cv2.applyColorMap(dispL, cv2.COLORMAP_HOT)
                cv2.imshow('img', imgL)
                cv2.imshow('disp', heatmapL)
                cv2.waitKey(1)

            if self.save_img:
                self.vid_writer.write(imgL)

            time.sleep(0.0001)


if __name__ == '__main__':
    det = detector(source='4', fps=10, saveImg=False, showImg=True, publish=False, client=False)
    det.det_process()
