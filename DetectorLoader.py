import time
import torch
import numpy as np
import torchvision.transforms as transforms

from queue import Queue
from threading import Thread

from Detection.Models import Darknet
from Detection.Utils import non_max_suppression, ResizePadding


class TinyYOLOv3_onecls(object):
    """载入训练好的Tiny-YOLOv3单人识别模型
    Args:
        input_size: (int)输入图像尺寸,必须是32的倍数
        config_file: (str) Yolo模型结构的config file路径
        weight_file: (str) 训练权重的文件路径
        nms: (float) 非极大值抑制nms的threshold
        conf_thres: 得分框重复度的阈值   #bbox->Bounding Box边界框
        device: (str) 跑模型的设备'cpu' or 'cuda'.
    """

    def __init__(self,
                 input_size=416,
                 config_file='Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg',
                 weight_file='Models/yolo-tiny-onecls/best-model.pth',
                 nms=0.2,
                 conf_thres=0.45,
                 device='cuda'):
        self.input_size = input_size
        self.model = Darknet(config_file).to(device)
        self.model.load_state_dict(torch.load(weight_file))
        self.model.eval()
        self.device = device

        self.nms = nms
        self.conf_thres = conf_thres

        self.resize_fn = ResizePadding(input_size, input_size)  # 图片压缩到指定大小
        self.transf_fn = transforms.ToTensor()  # [h,w,c]->[c,h,w] 并把0-255 压缩到0-1

    def detect(self, image, need_resize=True, expand_bb=5):
        """反馈给模型.
        Args:
            image: 需要检测的单张RGB图像(numpy array的形式) ,
            need_resize: 将输入的图片调整至指定大小(bool型) ,
            expand_bb: 拓展bbox的边框(int型).
        Returns:
            返回类型值为torch.float32
            每一个检测的目标都包含了下列指标[top, left, bottom, right, bbox_score, class_score, class]
            若未检测到，则返回‘None’
        """
        image_size = (self.input_size, self.input_size)
        if need_resize:
            image_size = image.shape[:2]
            image = self.resize_fn(image)

        image = self.transf_fn(image)[None, ...]
        scf = torch.min(self.input_size / torch.FloatTensor([image_size]), 1)[0]

        detected = self.model(image.to(self.device))
        detected = non_max_suppression(detected, self.conf_thres, self.nms)[0]
        if detected is not None:
            detected[:, [0, 2]] -= (self.input_size - scf * image_size[1]) / 2
            detected[:, [1, 3]] -= (self.input_size - scf * image_size[0]) / 2
            detected[:, 0:4] /= scf

            detected[:, 0:2] = np.maximum(0, detected[:, 0:2] - expand_bb)
            detected[:, 2:4] = np.minimum(image_size[::-1], detected[:, 2:4] + expand_bb)

        return detected


class ThreadDetection(object):
    def __init__(self,
                 dataloader,
                 model,
                 queue_size=256):
        self.model = model

        self.dataloader = dataloader
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)

    def start(self):
        t = Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            images = self.dataloader.getitem()

            outputs = self.model.detect(images)

            if self.Q.full():
                time.sleep(2)
            self.Q.put((images, outputs))

    def getitem(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True

    def __len__(self):
        return self.Q.qsize()