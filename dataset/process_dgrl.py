import struct
import os
import cv2 as cv
import numpy as np

def read_from_dgrl(dgrl):
    if not os.path.exists(dgrl):
        print('DGRL not exis!')
        return
    

    label_dir = dgrl+'_label'
    image_dir = dgrl+'_images'

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for p , base_name in enumerate(os.listdir(dgrl)):
        if base_name.endswith('.dgrl'):

            with open(dgrl + base_name, 'rb') as f:
                # 读取表头尺寸
                header_size = np.fromfile(f, dtype='uint8', count=4)
                header_size = sum([j<<(i*8) for i,j in enumerate(header_size)])
                # print(header_size)
                
                # 读取表头剩下内容，提取 code_length
                header = np.fromfile(f, dtype='uint8', count=header_size-4)
                code_length = sum([j<<(i*8) for i,j in enumerate(header[-4:-2])])
                # print(code_length)
                
                # 读取图像尺寸信息，提取图像中行数量
                image_record = np.fromfile(f, dtype='uint8', count=12)
                height = sum([j<<(i*8) for i,j in enumerate(image_record[:4])])
                width = sum([j<<(i*8) for i,j in enumerate(image_record[4:8])])
                line_num = sum([j<<(i*8) for i,j in enumerate(image_record[8:])])
                print('图像尺寸:')
                print(height, width, line_num)
                
                # 读取每一行的信息
                for k in range(line_num):
                    print(k+1)

                    # 读取该行的字符数量
                    char_num = np.fromfile(f, dtype='uint8', count=4)
                    char_num = sum([j<<(i*8) for i,j in enumerate(char_num)])
                    print('字符数量:', char_num)
                    
                    # 读取该行的标注信息
                    label = np.fromfile(f, dtype='uint8', count=code_length*char_num)
                    label = [label[i]<<(8*(i%code_length)) for i in range(code_length*char_num)]
                    label = [sum(label[i*code_length:(i+1)*code_length]) for i in range(char_num)]
                    label = [struct.pack('I', i).decode('gbk', 'ignore')[0] for i in label]
                    print('合并前：', label)
                    label = ''.join(label)
                    label = ''.join(label.split(b'\x00'.decode()))	# 去掉不可见字符 \x00，这一步不加的话后面保存的内容会出现看不见的问题
                    print('合并后：', label)
                    
                    # 读取该行的位置和尺寸
                    pos_size = np.fromfile(f, dtype='uint8', count=16)
                    y = sum([j<<(i*8) for i,j in enumerate(pos_size[:4])])
                    x = sum([j<<(i*8) for i,j in enumerate(pos_size[4:8])])
                    h = sum([j<<(i*8) for i,j in enumerate(pos_size[8:12])])
                    w = sum([j<<(i*8) for i,j in enumerate(pos_size[12:])])
                    # print(x, y, w, h)
                    
                    # 读取该行的图片
                    bitmap = np.fromfile(f, dtype='uint8', count=h*w)
                    bitmap = np.array(bitmap).reshape(h, w)
                    
                    # 保存信息
                    label_file = os.path.join(label_dir, base_name.replace('.dgrl', str(p)+'_'+str(k)+'.txt'))
                    with open(label_file, 'w') as f1:
                        f1.write(label)
                    bitmap_file = os.path.join(image_dir, base_name.replace('.dgrl', str(p)+'_'+str(k)+'.jpg'))
                    cv.imwrite(bitmap_file, bitmap)


read_from_dgrl('Competition13Line/')