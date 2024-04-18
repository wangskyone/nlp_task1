import math

from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# Load a model


#
model = YOLO('/nas_data/WTY/project/nlp_task1/runs/detect/train2/weights/best.pt')  # load a pretrained model (recommended for training)


# Train the model with 2 GPUs
# model = YOLO('/nas_data/WTY/project/nlp_task1/runs/detect/train2/weights/best.pt')
# results = model.train(data='custom.yaml', epochs=100, imgsz=640, device=[1,2])


# sort crop images
def regularization_sorting(Lx, Ly, W, H, alpha, beta):
    sorted_Lx, sorted_Ly, sorted_W, sorted_H = [], [], [], []

    while len(Lx):
        # Step 3: Calculate the average value M within the range of alpha for the minimum values of Ly
        min_Ly = min(Ly)
        M = sum(y for y in Ly if y <= min_Ly + alpha) / len([y for y in Ly if y <= min_Ly + alpha])

        # Step 4: Treat the index i of characters that are within a distance of beta from the mean M
        indices = [i for i, y in enumerate(Ly) if abs(y - M) <= beta]

        # Step 5: Sort i according to horizontal coordinate from small to large
        indices.sort(key=lambda i: Lx[i])

        # Step 6: Take sorted coordinates according to i into sorted_Lx, sorted_Ly, sorted_W, sorted_H
        for i in indices:
            sorted_Lx.append(Lx[i])
            sorted_Ly.append(Ly[i])
            sorted_W.append(W[i])
            sorted_H.append(H[i])

        # Step 7: Remove the coordinates already taken from Lx, Ly, W, H
        Lx = [x for i, x in enumerate(Lx) if i not in indices]
        Ly = [y for i, y in enumerate(Ly) if i not in indices]
        W = [w for i, w in enumerate(W) if i not in indices]
        H = [h for i, h in enumerate(H) if i not in indices]

    return sorted_Lx, sorted_Ly, sorted_W, sorted_H


# convert x y w h n to lx ly w h
def convert_boxes(boxes, size):
    for box in boxes:
        box[0] = box[0] * size[0]
        box[1] = box[1] * size[1]
        box[2] = box[2] * size[0]
        box[3] = box[3] * size[1]
        box[0] = box[0] - box[2] / 2
        box[1] = box[1] - box[3] / 2

    return boxes


# eg. crop and sort images
def get_img_text_dict(file_path=r'/nas_data/WTY/dataset/visualC3/train/'):
    char_label = file_path + 'char_label/'
    img_root_path = file_path + 'imgs/'
    img_path_text_dict = {}

    char_labels = ['100_0001.txt']
    for item in tqdm(char_labels):
        text_path = char_label + item
        f = open(text_path, mode='r', encoding='utf-8')
        texts = f.readlines()
        f.close()
        temp = []
        for tx in texts:
            tx = tx.strip()
            tx = tx.split(',')
            ch = tx[-1]
            x = float(tx[0])
            y = float(tx[1])
            w = float(tx[2])
            h = float(tx[3])
            temp.append([x,y,w,h,ch])
        img_path_text_dict[img_root_path + item.split('.')[0] + '.jpg'] = temp
    return img_path_text_dict


def crop_sort_img(file_path=r'/nas_data/WTY/dataset/visualC3/train/'):
    img_text_dict = get_img_text_dict(file_path=file_path)
    for key, value in tqdm(img_text_dict.items()):
        results = model.predict([key], conf=0.5)
        r = results[0]
        img = Image.open(key)
        img = img.convert('RGB')
        size = img.size
        boxes = r.boxes.xywhn.cpu().numpy()
        boxes = convert_boxes(boxes, size)
        
        alpha = boxes[:, 3].max()
        beta = alpha * 2 / 3
        sb1, sb2, sb3, sb4 = regularization_sorting(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], alpha=alpha,
                                                    beta=beta)
        sb1, sb2, sb3, sb4 = np.expand_dims(sb1, axis=1), np.expand_dims(sb2, axis=1), np.expand_dims(sb3,
                                                                                                      axis=1), np.expand_dims(
            sb4, axis=1)
        boxes = np.concatenate([sb1, sb2, sb3, sb4], axis=1)
        # change 'train' to 'valid' when get valid images index
        f = open(r'results/train/index_img.txt', mode='a+', encoding='utf-8')
        for i, box in enumerate(boxes):
            x, y, w, h = box
            value = sorted(value, key=lambda t: math.sqrt((t[0] - x) ** 2 + (t[1] - y) ** 2))
            max_distance = 5
            if math.sqrt((value[0][0] - x) ** 2 + (value[0][1] - y) ** 2) > max_distance:
                continue
            
            choose_ch = value[0][-1]

            crop = img.crop((x, y, x + w, y + h))
            # change 'train' to 'valid' when get valid images index
            crop.save(r'results/train/' + key.split('/')[-1].replace('.jpg', '_' + str(i) + '.jpg'))

            f.write(
                r'results/train/' + key.split('/')[-1].replace('.jpg', '_' + str(i) + '.jpg') + '\t' + choose_ch + '\n')

        f.close()


if __name__ == '__main__':
    crop_sort_img()
