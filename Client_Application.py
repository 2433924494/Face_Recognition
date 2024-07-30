import time
import requests
import cv2
import base64
import json
import argparse
from aip import AipFace
from threading import Thread
from apscheduler.schedulers.blocking import BlockingScheduler
import logging

with open('../Baidu_keys.json','r') as fp:
    keys=json.load(fp)
with open('../room_info.json','r') as fp:
    room_info=json.load(fp) 
imageType = "BASE64"
groupIdList=room_info['room_id']
options={
    'liveness_control':'NORMAL',
    'match_threshold':80
}   
logging.basicConfig(filename='./logs.txt',level=logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',encoding='utf-8')
PROTOTXT_PATH='./models/deploy_prototxt.txt'
MODEL_PATH='./models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
model = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)


def run():
    cap = cv2.VideoCapture(args.video_device)
    start_time=time.time()
    while True:
        if time.time()-start_time>15:
            break
        ret, frame = cap.read()
        if not ret:
            print('cap error,system exit')
            break
        # 反转图像
        frame = cv2.flip(frame, 1)
        face_exist,face_ROI=Get_Face(frame)
        if face_exist:
            response=detect_face(face_ROI)
            face_name=response['Name']
            If_Live=response['If_Live']
            # draw_name(face_detect,frame)
            print(f'姓名:{face_name} 活体:{True if If_Live==1 else False}')
            logging.info(f'姓名:{face_name} 活体:{True if If_Live==1 else False}')
        else:
            print('None')
            logging.warning('No faces!')
        # 展示视频画面
        cv2.imshow('video capture', frame)
        key = cv2.waitKey(10) & 0xff
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break



def req_baidu(client:AipFace,frame,):
    
    img_base64=image_to_base64(frame)
    response=client.multiSearch(img_base64,imageType,groupIdList,options)
    # print(response)
    
    if response["error_msg"]=='SUCCESS':
        result=response['result']
        face=result['face_list'][0]
        # if face:
        #     x=int(face["location"]['left'])
        #     y=int(face["location"]['top'])
        #     w=int(face["location"]['width'])
        #     h=int(face["location"]['height'])
        #     cv2.rectangle(frame,(x,y),(x+w,y+h),color=(255,0,0),thickness=2)
        print(f"Name:{face['user_list'][0]['user_info']} Score:{face['user_list'][0]['score']:.2f}")
        logging.info(f"Name:{face['user_list'][0]['user_info']}")
        return True
    else:
        # print('Name:Unknown')
        logging.warning(f'Request API failed! Erro:{response["error_msg"]}')
        return False
def run_Baidu():
    cap=cv2.VideoCapture(args.video_device)
    
    client=AipFace(keys['APP_ID'],keys['API_KEY'],keys['SECRET_KEY'])
    
    # camara_display(cap=cap)
    start_time=time.time()
    while True:
        if time.time()-start_time>15:
            break
        ret,frame=cap.read()
        frame = cv2.flip(frame, 1)
        draw_face(frame,dnn_face_detect(frame))
        if req_baidu(client=client,frame=frame):
            break
        # 测试用
        # time.sleep(1)
        cv2.imshow('video capture', frame)
        key = cv2.waitKey(10) & 0xff
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
    
import numpy as np
def get_ROI(x,y,w,h,frame):
    block_len=max(w,h)
    # ROI=np.zeros((block_len,block_len,3),np.uint8)
    x1=(x+w) if ((x+w)<frame.shape[1]) else (frame.shape[1])
    y1=(y+h) if ((y+h)<frame.shape[0]) else (frame.shape[0])
    ROI=frame[y:y1-1,x:x1-1]
    ROI=np.pad(ROI,((0,block_len-(y1-y)),(0,block_len-(x1-x)),(0,0)),'constant')
    # print(f'ROI_point:{x,y,x1-1,y1-1}  Rec+point{x,y,x+w,y+h}')
    cv2.imshow('ROI', ROI)
    return ROI

def dnn_face_detect(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
    model.setInput(blob)
    faces=np.squeeze(model.forward())

    return faces

def Get_Face(frame):
    ori_img=frame.copy()
    h,w=frame.shape[:2]
    faces=dnn_face_detect(frame)
    # 为人脸画框
    draw_face(faces=faces,frame=frame)
    max_len=0
    max_index=-1
    for i in range(faces.shape[0]):
        confidence=faces[i,2]
        if confidence>0.5:
            start_x, start_y, end_x, end_y=faces[i,3:7]
            len=max(end_x-start_x,end_y-start_y)
            if len>max_len:
                max_index=i
    box = faces[max_index, 3:7] * np.array([w, h, w, h])
    start_x, start_y, end_x, end_y = box.astype(int)
    Face_pos=[start_x,start_y,end_x-start_x+1,end_y-start_y+1]
    
    if max_index>-1:
        return True,get_ROI(*(Face_pos),ori_img)
    else:
        return False,None
def draw_name(name,frame,pos):
    # pilimage=Image.fromarray(frame)
    # fontpath = "simsun.ttc"
    pass

def draw_face(frame,faces):
    h,w=frame.shape[:2]
    for i in range(0, faces.shape[0]):
        confidence = faces[i, 2]
        if confidence>0.5:
            box = faces[i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype(int)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
    
def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code

def detect_face(face):
    face = cv2.resize(face, (160, 160))
    img=image_to_base64(face)
    files={'img':img}
    response=requests.post('http://127.0.0.1:8000/detect_face',json.dumps(files))
    return response.json()


parser=argparse.ArgumentParser()
parser.add_argument('-m',dest='mode',type=int,default=1,help='0/1 Baidu-api or Server-api')
parser.add_argument('-vd',dest='video_device',type=int,default=0,help='video device number')
args=parser.parse_args()
if __name__ == '__main__':
    
    if args.mode:
        run()
    else:
        run_Baidu()