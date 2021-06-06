from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync

warnings.filterwarnings('ignore')

import os
import csv
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QMessageBox
from ui.Ui_start import Ui_VV
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from TTS.TTS import TTS
import random
from getCommandIndex import CommandIndex
required_classes_from_yolo =['car','truck','bus','person','bicycle','motorbike']


class App:

    def __init__(self):
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 1.0

        self.model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)

        self.tracking = True
        self.writeVideo_flag = True
        self.asyncVideo_flag = False
        self.camera_flag = False
        self.track_cmd_list= {}
        self.current_track_id = ""
        self.skip_frame = 0
        self.frame_counter = 0
        self.filename = "sample_video/dattatraya"
        self.extension = ".mp4"
        self.file_path = self.filename+self.extension

        self.yolo = YOLO()

        self.MOTres = []

        self.main = QMainWindow()
        self.ui = Ui_VV()
        self.ui.setupUi(self.main)

        # create video capture

        if self.asyncVideo_flag :
            self.video_capture = VideoCaptureAsync(self.file_path)
        else:
            if self.camera_flag :
                self.video_capture = cv2.VideoCapture('http://192.168.1.103:8080')
            else:
                self.video_capture = cv2.VideoCapture(self.file_path)

        # self.cap = cv2.VideoCapture(0)
        self.thread = TTS()
        self.thread.start()
        self.thread.setLanguage(False)
        
        # create a timer
        self.timer = QTimer()

        # connect signal to slot        
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # self.timer.timeout.connect(self.videoChoose)

        # set control_bt callback clicked  function
        # self.ui.screenshot.clicked.connect(self.controlTimer)
        
        self.ui.exit.clicked.connect(self.exit)
        # self.ui.screenshot.clicked.connect(self.screenshot)

    def viewCam(self):

        if self.asyncVideo_flag:
            self.video_capture.start()

        if self.writeVideo_flag:
            if self.asyncVideo_flag:
                self.w = int(self.video_capture.cap.get(3))
                self.h = int(self.video_capture.cap.get(4))
            else:
                self.w = int(self.video_capture.get(3))
                self.h = int(self.video_capture.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (self.w, self.h))
            self.frame_index = -1

        self.fps = 0.0
        self.fps_imutils = imutils.video.FPS().start()
        

        
        ret, frame = self.video_capture.read()  # frame shape 640*480*3
        if ret != True:
            return 0
        elif self.skip_frame:
            self.skip_frame-=1
            return 0
        else:
            self.skip_frame = 5
        t1 = time.time()
        self.frame_counter+=1

        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxes, confidence, classes = self.yolo.detect_image(image)

        if self.tracking:
            features = self.encoder(frame, boxes)
            detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                          zip(boxes, confidence, classes, features)]
        else:
            detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                          zip(boxes, confidence, classes)]
       
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        detections_result = np.copy(frame)
        if self.tracking:
            # Call the tracker
            self.tracker.predict()
            self.tracker.update(detections)

            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                self.MOTres.append([self.frame_counter,track.track_id,bbox[0],bbox[1],bbox[2],bbox[3],1,1,1,1])
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, str(track.cls) + " : " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                            1.10e-3 * frame.shape[0], (0, 255, 0), 1)
                classIndex=CommandIndex(frame,bbox,required_classes_from_yolo.index(track.cls))
                idx = classIndex.returnIndex()
                try:
                    if idx != self.track_cmd_list.get(track.track_id,0):
                        self.thread.addCommand(idx)
                        self.track_cmd_list[track.track_id] = idx
                except:
                    pass
        for det in detections:
            bbox = det.to_tlbr()
            score = "%.2f" % round(det.confidence * 100, 2) + "%"
            cv2.rectangle(detections_result, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            if len(classes) > 0:
                cls = det.cls
                cv2.putText(detections_result, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                            1.5e-3 * detections_result.shape[0], (0, 255, 0), 1)        
        cv2.putText(detections_result, "Frame no: " + str(self.frame_counter), (20,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2,cv2.LINE_AA)
        assert cv2.imwrite(str(self.frame_counter) + ".jpg",detections_result)
        if self.writeVideo_flag: # and not asyncVideo_flag:
            # save a frame
            self.out.write(frame)
            self.frame_index = self.frame_index + 1
        
        self.fps_imutils.update()

        if not self.asyncVideo_flag:
            self.fps = (self.fps + (1./(time.time()-t1))) / 2
            print("FPS = %f"%(self.fps))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get image infos
        height, width, channel = frame.shape
        step = channel * width

        # create QImage from image
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)

        # show image in img_label
        self.ui.DCTpreview.setPixmap(QPixmap.fromImage(qImg))

    def show(self):
        # self.main_window.show()
        self.main.show()
        self.controlTimer()

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # start timer
            self.timer.start(20)

            # update control_bt text
            # self.ui.screenshot.setText("Screen stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()

            # release video capture
            self.video_capture.release()

            # update control_bt text
            # self.ui.screenshot.setText("Screen start")
    
    # for exiting app
    def exit(self):
        self.fps_imutils.stop()
        self.thread.exitTTS()
        print('imutils FPS: {}'.format(self.fps_imutils.fps()))
        if not os.path.exists('MOT/'+self.filename):
            os.mkdir('MOT/'+self.filename)
        with open('MOT/'+self.filename+'/res.txt', 'w') as f: 
            write = csv.writer(f) 
            write.writerows(self.MOTres) 
        
        if self.asyncVideo_flag:
            self.video_capture.stop()
        else:
            self.video_capture.release()

        if self.writeVideo_flag:
            self.out.release()

        sys.exit()
        # self.thread.command_list = []

#Entry point
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = App()
    main.show()
    sys.exit(app.exec_())
