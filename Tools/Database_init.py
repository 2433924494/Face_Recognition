import sqlite3

def db_init():
    conn=sqlite3.connect('./Face_interface_database/face_interfaces.db')
    cur=conn.cursor()
    sql_codes='''create table faces
                (face_token TEXT,
                name TEXT,
                name_id TEXT,
                group_id TEXT,
                img_base64 TEXT);
                '''
    cur.execute(sql_codes)
    conn.commit()
    cur.close()
    conn.close()

if __name__ == '__main__':
    db_init()