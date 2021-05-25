from __future__ import print_function
import os
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QImage
from GUI import *
import sys
from threading import Thread
import win32gui
import time
import core.utils as utils
import tensorflow as tf
from PIL import Image
from datetime import datetime
from hog import *

class QTypeSignal(QMainWindow):
    #定义一个信号
    # sendmsg = pyqtSignal(object)
    sendmsg=pyqtSignal(str)
    def __init__(self):
        super(QTypeSignal, self).__init__()
    def run(self,msg1):
        #发射信号
        #self.sendmsg.emit()
        self.sendmsg.emit(msg1)

class QTypeSlot(QMainWindow):
    def __init__(self):
        super(QTypeSlot, self).__init__()
    #槽对象中的槽函数
    # def get( self,msg ):
    #     print("QSlot get msg => " + msg)
    def get(self,msg1):
        QMessageBox.about(self, '检测结束', msg1+" 已经检测完成")
        # reply2 = QMessageBox.information(self, '检测结束', '是否播放检测结果',
        #                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # if reply2 == QMessageBox.Yes:
        #     print (msg1)
            # self.player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))  # 选取视频文件
            # self.player.play()  # 播放视频

class QTypeSignal2(QMainWindow):
    sendmsg2=pyqtSignal(str)
    def __init__(self):
        super(QTypeSignal2, self).__init__()
    def run(self,msg1):
        self.sendmsg2.emit(msg1)

class QTypeSlot2(QMainWindow):
    def __init__(self):
        super(QTypeSlot2, self).__init__()
    def get(self,msg1):
        QMessageBox.about(self, '录屏结束', msg1+" 已经录制完成")

class myMainWindow(Ui_MainWindow,QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.wgt_video)  # 视频播放输出
        self.btn_open.clicked.connect(self.openVideoFile)   # 打开视频文件按钮
        self.btn_play.clicked.connect(self.playVideo)       # play
        self.btn_stop.clicked.connect(self.pauseVideo)       # pause
        self.player.positionChanged.connect(self.changeSlide)      # change Slide
        self.sld_video.sliderPressed.connect(self.lianjie)
        self.sld_video.sliderReleased.connect(self.duankai)
        self.btn_detect.clicked.connect(self.detectVideoFile)  # 检测视频文件按钮
        self.btn_yolo.clicked.connect(self.detectYolo)
        self.btn_jietu.clicked.connect(self.jietu)
        self.btn_lupin.clicked.connect(self.lupin)
        self.btn_stop.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.send = QTypeSignal()
        self.slot = QTypeSlot()
        self.send.sendmsg.connect(self.slot.get)
        self.send2 = QTypeSignal2()
        self.slot2 = QTypeSlot2()
        self.send2.sendmsg2.connect(self.slot2.get)

    def QImageToCvMat(self,incomingImage):
        incomingImage = incomingImage.convertToFormat(QImage.Format.Format_RGBA8888)
        width = incomingImage.width()
        height = incomingImage.height()
        ptr = incomingImage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return arr

    def screenvideo(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        hwnd = win32gui.FindWindow(None, 'MainWindow')
        screen = QApplication.primaryScreen()
        img = screen.grabWindow(hwnd).toImage()
        width = img.width()
        height = img.height()
        localtime = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        output_movie = cv2.VideoWriter(localtime + '.mp4', fourcc, 25, (int(width), int(height)))
        framecount = 0
        framenumber = self.lupintime.value()*25
        while framecount != framenumber:
            hwnd = win32gui.FindWindow(None, 'MainWindow')
            screen = QApplication.primaryScreen()
            img = screen.grabWindow(hwnd).toImage()
            #img = screen.grabWindow(hwnd).toMat()
            img.save("temp.jpg")
            imgread = cv2.imread("temp.jpg")
            output_movie.write(imgread)
            framecount = framecount + 1
        output_movie.release()
        self.send2.run(localtime + '.mp4')

    def lupin(self):
        t4 = Thread(target=self.screenvideo)
        t4.start()

    def detectYolo(self):
        file = QFileDialog.getOpenFileName(self, '选择一个需要进行检测的视频', './',
                                           'video(*.mp4 *.avi *.flv *.mkv *.mpg)')[0]
        print(file)
        t2 = Thread(target=self.yolo, args=(file,))
        t2.start()

    def yolo(self,video_path):
        return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0",
                           "pred_lbbox/concat_2:0"]
        pb_file = "./yolov3_coco.pb"
        num_classes = 80
        input_size = 416
        graph = tf.Graph()
        return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
        video_name_path, _ = video_path.split('.')
        video_file_path = os.path.dirname(video_path)
        video_name = video_name_path[len(video_file_path) + 1:]
        print(video_name)

        tstart = datetime.now()
        with tf.Session(graph=graph) as sess:
            vid = cv2.VideoCapture(video_path)
            length = vid.get(5)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = vid.get(3)
            height = vid.get(4)
            print(width, height)
            output_movie = cv2.VideoWriter('YOLO' + video_name + '.mp4', fourcc, length, (int(width), int(height)))
            return_value = True
            while (return_value):
                return_value, frame = vid.read()
                if return_value == False:
                    break
                print('Read a new frame:', return_value)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                frame_size = frame.shape[:2]
                image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
                image_data = image_data[np.newaxis, ...]
                prev_time = time.time()

                pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                    [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={return_tensors[0]: image_data})

                pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

                bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
                bboxes = utils.nms(bboxes, 0.45, method='nms')
                image = utils.draw_bbox(frame, bboxes)

                curr_time = time.time()
                exec_time = curr_time - prev_time
                result = np.asarray(image)
                info = "time: %.2f ms" % (1000 * exec_time)
                result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                output_movie.write(result)
            vid.release()
            output_movie.release()

        tend = datetime.now()
        print("Time cost = ", (tend - tstart))
        self.send.run(video_name + '.mp4')

    def duankai(self):
        self.sld_video.valueChanged[int].disconnect(self.changeValue)

    def lianjie(self):
        self.sld_video.valueChanged[int].connect(self.changeValue)

    def onTimerOut(self):
        self.vidoeLength = self.player.duration() + 0.1
        self.sld_video.setValue(round((self.player.position()/ self.vidoeLength) * 100))
        self.lab_video.setText(str(round((self.player.position() / self.vidoeLength) * 100, 2)) + '%')

    def openVideoFile(self):
        self.player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))  # 选取视频文件
        self.player.play()  # 播放视频
        self.btn_stop.setEnabled(True)

    def screenshot(self):
        hwnd = win32gui.FindWindow(None, 'MainWindow')
        screen = QApplication.primaryScreen()
        img = screen.grabWindow(hwnd).toImage()
        localtime = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        print(localtime + '.jpg')
        localtime = localtime+'.jpg'
        localtime = str(localtime)
        boolsave = img.save(localtime,"JPG",100)
        if(boolsave):
            # print (localtime)
            print (" success")

    def jietu(self):
        t3 = Thread(target=self.screenshot)
        t3.start()
        # pix = QScreen.grabWindow(self,0,0,0,300,300)
        # filename = QFileDialog.getSaveFileName(self, "保存图片", "./",
        #                                        "PNG(*.png);;JPG(*.jpg);;BMP(*.bmp)")
        # #print(filename)
        # pix.save(self,filename+".png")
        # hwnd = win32gui.FindWindow("MainWindow")
        # win32gui.ShowWindow(hwnd,win32con.SW_RESTORE)
        # win32gui.SetForegroundWindow(hwnd)
        # game_rect = win32gui.GetWindowRect(hwnd)
        # src_image = ImageGrab.grab(game_rect)
        # filename = QFileDialog.getSaveFileName(self,"保存图片","./",
        #                                        "PNG(*.png);;JPG(*.jpg);;BMP(*.bmp)")
        # src_image.save(self,filename)

    def detectVideoFile(self):
        file = QFileDialog.getOpenFileName(self,'选择一个需要进行检测的视频','./',
                                           'video(*.mp4 *.avi *.flv *.mkv *.mpg)')[0]
        print(file)
        t1 = Thread(target=self.v2i,args=(file,))
        t1.start()
        #QMessageBox.about(self, '检测结束', "视频检测完成")
        # self.thread = WorkThread()
        # self.thread.start();
        #self.v2i(file)
        #QApplication.processEvents()

    def playVideo(self):
        self.player.play()
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def pauseVideo(self):
        self.player.pause()
        self.btn_stop.setEnabled(False)
        self.btn_play.setEnabled(True)

    def changeSlide(self,position):
        self.vidoeLength = self.player.duration()+0.1
        self.sld_video.setValue(round((position/self.vidoeLength)*100))
        self.lab_video.setText(str(round((position/self.vidoeLength)*100,2))+'%')

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '退出',"确定要退出嘛?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def changeValue(self):
        self.player.setPosition(self.sld_video.value()*self.player.duration()/self.sld_video.maximum())

    def v2i(self,videos):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print(videos)
        tstart = datetime.now()
        video_name_path, _ = videos.split('.')
        video_file_path = os.path.dirname(videos)
        video_name = video_name_path[len(video_file_path) + 1:]
        print(video_name)

        cap = cv2.VideoCapture(videos)
        frame_count = 1
        success = True
        length = cap.get(5)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = cap.get(3)
        height = cap.get(4)
        print(width, height)
        output_movie = cv2.VideoWriter( 'HOG'+video_name+'.mp4', fourcc, length, (int(width), int(height)))
        model_path = 'svm_model'
        while (success):
            success, frame = cap.read()
            print('Read a new frame:', success)
            if (success == False):
                break;
            frame = imutils.resize(frame, width=min(400, frame.shape[1]))

            # if frame_count%2==0:
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                   padding=(8, 8), scale=1.5)

            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            #rects = np.array([[x, y, x + w, y + h] for (x, y, _,w, h) in detections])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            # draw the final bounding boxes
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(frame, (xA+10, yA+10), (xB-10, yB-10), (0, 255, 0), 2)
            output_movie.write(frame)
            frame_count = frame_count + 1
        cap.release()
        output_movie.release()
        tend = datetime.now()
        print("Time cost = ", (tend - tstart))
        self.send.run(video_name+'.mp4')
        #QMessageBox.about(self, '检测结束', "视频检测完成")
        #time.sleep(10)

        # QMessageBox.information(self, '检测结束', '是否播放检测结果',
        #                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # if reply2 == QMessageBox.Yes:
        #     self.player.setMedia(output_movie)  # 选取视频文件
        #     self.player.play()  # 播放视频

if __name__ == '__main__':
    app = QApplication(sys.argv)
    vieo_gui = myMainWindow()
    vieo_gui.show()
    sys.exit(app.exec_())