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
from utils.rosutils import rosPublisher, rosClient, rosImgPublisher

LOGGER = set_logging(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

RKNN_MODEL = 'yolov5n_coco_VOC_mower_416_final.rknn'


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
        self.img_pub = rosImgPublisher(self.fps)

        if self.save_img:
            save_path = str(ROOT / 'demo.mp4')
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.vid_writer = cv2.VideoWriter(save_path, fourcc, 6, self.imgsz)

        if self.publish:
            self.pub = rosPublisher(self.fps)

        if self.client:
            # set dist_min for danger state
            self.dist_min = 1000
            self.safe_count = 0
            #self.check_count = 0
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

        self.input_q_yolov5 = Queue(maxsize=1)
        self.output_q_yolov5 = Queue(maxsize=1)
        self.input_q_sgbm = Queue(maxsize=1)
        self.output_q_sgbm = Queue(maxsize=1)

        # time recorder
        self.dt, self.seen = [0.0, 0.0, 0.0], 0

    def build_threads(self):
        self.thread4dataset = Thread(target=self.dataset_queue, daemon=True)
        self.thread4yolov5 = Thread(target=self.yolov5_process, daemon=True)
        self.thread4sgbm = Thread(target=self.sgbm_process, daemon=True)

        self.thread4dataset.start()
        self.thread4yolov5.start()
        self.thread4sgbm.start()

    def dataset_queue(self):
       for imgL, imgR in self.dataset:
            # letterbox
            imgL, _, _ = letterbox(imgL, self.imgsz)
            imgR, _, _ = letterbox(imgR, self.imgsz)
            # if one queue is full, sleep to wait process
            while self.input_q_yolov5.full() or self.input_q_sgbm.full():
                time.sleep(0.0001)
            self.input_q_yolov5.put([imgL, imgR])
            self.input_q_sgbm.put([imgL, imgR])
            time.sleep(0.0001)

    def yolov5_process(self):
        while True:
            # only if input_queue is not empty, output_queue is not full, then process
            if not self.input_q_yolov5.empty() and not self.output_q_yolov5.full():
                t = time.time()
                # First-stage: yolov5 prediction
                imgL, imgR = self.input_q_yolov5.get()
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

                self.output_q_yolov5.put([boxes, classes, scores])
                self.dt[0] = time.time() - t
            time.sleep(0.0001)

    def sgbm_process(self):
        while True:
            # only if input_queue is not empty, output_queue is not full, then process
            if not self.input_q_sgbm.empty() and not self.output_q_sgbm.full():
                t = time.time()
                # Second-stage: SGBM depth estimation
                imgL, imgR = self.input_q_sgbm.get()
                # BGR2GRAY
                imL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
                imR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
                # estimation
                dispL, coord_3d = self.img_processor.stereoMatchSGBM(imL, imR, down_scale=True)
                self.output_q_sgbm.put([imgL, dispL, coord_3d])
                self.dt[1] = time.time() - t
            time.sleep(0.0001)
 
    def det_process(self):
        self.build_threads()

        while True:
            self.seen += 1
            t = time.time()

            # Third-stage: distance extraction
            # if one queue is empty, sleep to wait process
            while self.output_q_yolov5.empty() or self.output_q_sgbm.empty():
                time.sleep(0.0001)
            boxes, classes, scores = self.output_q_yolov5.get()
            imgL, dispL, coord_3d = self.output_q_sgbm.get()

            dists = list()
            if boxes is not None:
                for box in boxes:
                    dist = obtain_depth(coord_3d, box, ratio=0.8)
                    # dists.append(dist)
                    # only publish dangerous distance
                    if dist < 0.6: dists.append(dist)
                if dists:
                    # draw boxes in img
                    draw(imgL, boxes, scores, classes, dists)
            # publish
            if self.publish:
                try:
                    if dists:
                        self.pub.pub_dist(np.amin(dists))
                    # test publish rate
                    #else:
                    #    self.pub.pub_dist(100000)
                except rospy.ROSInterruptException:
                    pass
            # send req
            if self.client:
                if dists:
                    # in danger state
                    # reset check_count in danger state
                    #self.check_count = 0
                    dist_temp = np.amin(dists)
                    if dist_temp < self.dist_min:
                        # if the obstacle is closer than before
                        # send req
                        self.dist_min = dist_temp
                        self.c.run(self.dist_min)
                        self.safe_count = 0
                else:
                    if self.dist_min != 1000:
                        # from danger state to safe state
                        self.safe_count += 1
                        if self.safe_count >= 10:
                            self.c.run(-1)
                            self.safe_count = 0
                            self.dist_min = 1000
                    #else:
                        # in safe state accumulate check_count
                        # every 20 frames check one time
                        #self.check_count += 1
                        #if self.check_count >= 20:
                            #self.c.run(-1)
                            # finish check, reset check_count
                            #self.check_count = 0
                    # else, do nothing
                    # test
                    #self.dist_min = 1000
                    #self.c.run(-1)

            self.dt[2] = time.time() - t

            LOGGER.info(f"yolov5: {self.dt[0]}, sgbm: {self.dt[1]}, distance: {self.dt[1]}")

            # show img
            if self.show_img:
                dispL = dispL.astype(np.uint8)
                heatmapL = cv2.applyColorMap(dispL, cv2.COLORMAP_HOT)
                cv2.imshow('img', imgL)
                cv2.imshow('disp', heatmapL)
                if cv2.waitKey(1) == ord('q'):
                    break

            if self.save_img:
                self.vid_writer.write(imgL)

            self.img_pub.pub_img(imgL)

            time.sleep(0.0001)
        self.vid_writer.release()


if __name__ == '__main__':
    rospy.init_node('detector', anonymous=True)
    det = detector(source='10', fps=6, imgSize=(416, 312), saveImg=False, showImg=False, publish=False, client=True)
    det.det_process()
