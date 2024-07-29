import time
import requests
import cv2
import base64
import json
import argparse
from aip import AipFace

PROTOTXT_PATH='./deploy_prototxt.txt'
MODEL_PATH='./models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
model = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
def run():
    cap = cv2.VideoCapture(args.video_device)
    while True:
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
        else:
            print('None')
        # 展示视频画面
        cv2.imshow('video capture', frame)
        key = cv2.waitKey(10) & 0xff
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break




def run_Baidu():
    cap=cv2.VideoCapture(args.video_device)
    client=AipFace(APP_ID, API_KEY, SECRET_KEY)
    imageType = "BASE64"
    groupIdList=args.room
    options={
        'liveness_control':'NORMAL',
        'match_threshold':70
    }   
    while True:
        ret,frame=cap.read()
        frame = cv2.flip(frame, 1)
        # face_exist,face_ROI=Get_Face(frame)
        
        img_base64=image_to_base64(frame)
        response=client.multiSearch(img_base64,imageType,groupIdList,options)
        # print(response)
        
        if response["error_msg"]=='SUCCESS':
            result=response['result']
            face=result['face_list'][0]
            if face:
                x=int(face["location"]['left'])
                y=int(face["location"]['top'])
                w=int(face["location"]['width'])
                h=int(face["location"]['height'])
                cv2.rectangle(frame,(x,y),(x+w,y+h),color=(255,0,0),thickness=2)
                print(f"Name:{face['user_list'][0]['user_info']} Score:{face['user_list'][0]['score']:.2f}")
            else:
                print('Name:Unknown')
        else:
            print(f'Request API failed!\nErro:{response["error_msg"]}')
        cv2.imshow('video capture', frame)
        time.sleep(1)
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
def Get_Face(frame):
    # face_detector = cv2.CascadeClassifier('./face_feature_detect/haarcascade_frontalface_default.xml')
    # grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_detector.detectMultiScale(grey)
    # print('Exist' if faces.any() else 'No faces')
    h,w=frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
    model.setInput(blob)
    faces=np.squeeze(model.forward())
    Max_face_len=-1
    Max_face_index=-1
    ori_img=frame.copy()
    Face_pos=[]
    cnt=-1
    for i in range(0, faces.shape[0]):
        confidence = faces[i, 2]
        if confidence > 0.5:
            cnt+=1
            # print(faces[i, 3:7].shape)
            # print(f'w:{w}h:{h}')
            # print(np.array([w, h, w, h]).shape)
            box = faces[i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype(int)
            Face_pos.append([start_x,start_y,end_x-start_x+1,end_y-start_y+1])
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
            if max((end_x-start_x),(end_y-start_y))>Max_face_len:
                Max_face_len=max((end_x-start_x),(end_y-start_y))
                Max_face_index=cnt
    # for i,(x, y, w, h) in enumerate(faces):
	# # 在人脸区域绘制矩形
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
    #     # ROI.append(get_ROI(x,y,w,h,frame))
    #     if max(w,h)>Max_face_len:
    #         Max_face_len=max(w,h)
    #         Max_face_index=i
    if Max_face_index>-1:
        return True,get_ROI(*(Face_pos[Max_face_index]),ori_img)
    else:
        return False,None
def draw_name(name,frame,pos):
    # pilimage=Image.fromarray(frame)
    # fontpath = "simsun.ttc"
    pass

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
parser.add_argument('-r',dest='room',type=str,help='room id e.g. 08_1803B ->(栋号)_(寝室号)')
args=parser.parse_args()
if __name__ == '__main__':
    if args.mode:
        run()
    else:
        run_Baidu()