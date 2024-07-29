import os
import time
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.facenet import Facenet
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import json
import uvicorn
import cv2
import numpy as np
import io
import base64
import Live_Detection.detect as LD
from iresnet import iresnet50

app = FastAPI()
device = select_device('')
half = device.type != 'cpu'  # half precision only supported on CUDA


def base64_to_image(base64_code):
    # base64解码
    img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.fromstring(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)

    return img


class Image(BaseModel):
    img: str


# net = iresnet50()
# net.load_state_dict(torch.load('./backbone.pth',map_location=torch.device('cpu')))
net=Facenet('mobilenet',mode='predict').eval()
net.load_state_dict(torch.load('./models/facenet_mobilenet.pth',map_location=torch.device('cpu')),strict=False)
net.eval()
# 获取人脸特征
@torch.no_grad()
def inference(img):
    # img 是人脸的剪切图像
    img = cv2.resize(img, (160, 160))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    feat = net(img).numpy().flatten()
    # print(feat)
    return feat

# 加载存储的人脸特征
def load_face_feature(dir):
    # 将人脸文件夹中的人脸特征都存到字典里，方便比对
    face_list = os.listdir(dir)
    # print(face_list)
    face_feature_dict = {}
    for face in face_list:
        img0 = cv2.imread(os.path.join(dir, face))
        # cv2.imshow('tee',img0)
        img0_feature = inference(img0)
        face_feature_dict[face.replace('.jpg', '')] = img0_feature
    return face_feature_dict
face_feature_dict = load_face_feature('./face_img_database')

# 人脸特征对比
def cosin_metric(x1, x2):
    # single feature vs. single feature
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


# 对比人脸特征返回比对名称，不存在返回none
def compare_face(face_img0, face_feature_dict):
    face_img0_feature = inference(face_img0)
    # print(face_img0_feature)
    max_prob = 0
    max_name = ''
    for name in face_feature_dict.keys():
        face_img1_feature = face_feature_dict[name]
        prob = cosin_metric(face_img0_feature, face_img1_feature)
        if prob > max_prob:
            max_prob = prob
            max_name = name
    # print(max_name, max_prob)

    if max_prob > 0.3:
        return max_name
    else:
        return 'unknown'



@app.post('/detect_face')
def detect_face(image:Image):
    img=base64_to_image(image.img)
    flag=compare_face(img,face_feature_dict)
    label=LD.Live_Detect(cv2.resize(img, (80, 80)),'./models/80x80_MiniFASNetV2.pth')
    return {'Name':flag,'If_Live':label}

if __name__ == '__main__':
    # fast:app 中的 fast=运行的文件名,如果修改了记得这里别忘记改
    uvicorn.run("Server_Application:app", host="0.0.0.0", port=8000, reload=True)
    # pred_list = detect(weights='./runs/train/exp8/weights/best.pt', source="/home/zk/git_projects/hand_pose/hand_pose_yolov5_5.0/hand_pose/images/four_fingers10.jpg")
    # print(pred_list)