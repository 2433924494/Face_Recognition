from aip import AipFace
import json
import argparse
# from Client_Application import image_to_base64
import cv2
import base64
import sqlite3

def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code

parse=argparse.ArgumentParser()
parse.add_argument('-i',dest='image',type=str,help='image path')
parse.add_argument('-g',dest='group',type=str,help='group id (0-9a-zA-Z_)')
parse.add_argument('-ui',dest='user_id',type=str,help='user id (0-9a-zA-Z_)')
parse.add_argument('-un',dest='user_name',type=str,help='user name')
args=parse.parse_args()

def add_face():
    with open('../Baidu_keys.json','r') as fp:
        keys=json.load(fp)
    client=AipFace(keys['APP_ID'],keys['API_KEY'],keys['SECRET_KEY'])
    image=image_to_base64(cv2.imread(args.image))
    group_id=args.group
    user_id=args.user_id

    options={
        'user_info':args.user_name,
        'action_type':'REPLACE'
    }

    result=client.addUser(image=image,
                image_type='BASE64',
                group_id=args.group,
                user_id=user_id,
                options=options)
    if result['error_msg']=='SUCCESS':
        print('Success!')
    else:
        print(f"Add face failed!\n error code:{result['error_code']}")
    return result,image
def record_face(result,image_base64):
    conn=sqlite3.connect('./face_interface_database/face_interfaces.db')
    cur=conn.cursor()
    sql_codes=f'''
                INSERT INTO faces VALUES('{result['result']['face_token']}',
                                        '{args.user_name}',
                                        '{args.user_id}',
                                        '{args.group}',
                                        '{image_base64}')
            '''
    cur.execute(sql_codes)
    conn.commit()
    cur.close()
    conn.close()
if __name__=='__main__':
    result,img=add_face()
    record_face(result,img)
# print(result)
