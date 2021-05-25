# videomanage-objectdetection
video management and object detection
# 视频数据的信息管理及目标检测系统
## 背景
  随着计算机科学技术与人工智能的持续发展，人们已经愈来愈关注目标检测研究方向。目前目标检测已经十分广泛地应用在无人驾驶、监控系统、智能交通、刑侦系统等许多领域。目标检测的主要内容是将我们不关心的背景信息过滤，去除背景信息的过程中采用了一系列的算法，之后提取出所关心的目标信息。当前，目标检测方法大致分为基于计算机视觉的传统方法和现在流行的深度学习方法。
## 系统简介
本系统主要是实现视频数据播放、暂停、拖动进度条等基本功能，能够保存视频以及对视频数据进行管理。可以通过基于HOG的目标检测和基于YOLO的目标检测对视频中的多个行人目标进行检测，并用矩形框标识目标，同时可以保存目标定位截图或对目标检测的视频片段进行录制并保存，实现对视频数据的信息管理以及对视频进行目标检测。
## requirements
```# packages in environment:
Keras                     2.2.4                   
lxml                      3.6.4         
matplotlib                1.5.3             
numpy                     1.15.1            
opencv-contrib-python     3.4.2.16   
opencv-python             3.4.4.19             
pandas                    0.18.1        
Pillow                    5.3.0                 
PyQt5                     5.13.0            
PyQt5-sip                 12.7.2              
pyqt5-tools               5.13.0.1.5          
python                    3.5.2         
qt                        5.6.0         
scikit-image              0.12.3           
scikit-learn              0.19.0          
scipy                     1.1.0               
tensorboard               1.11.0                 
tensorflow                1.11.0              
tensorflow-gpu            1.11.0            
tensorflow-tensorboard    0.4.0                 
xlrd                      1.0.0              
```
