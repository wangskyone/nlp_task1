import sys
sys.path.append('/nas_data/WTY/project/nlp_task1/clip_demo')
from ultralytics import YOLO
from clip_demo.cn_clip.clip import load_from_name
from clip_demo.cn_clip import clip
import torch
from PIL import Image
from train_yolo_1 import regularization_sorting, convert_boxes
import numpy as np
from clip_demo.utils import image_transform
from tqdm import tqdm
import os


def load_model_and_process(yolo_path, clip_head, clip_path, device):
    """
    加载模型和处理函数
    :param yolo_path: yolo模型路径
    :param clip_head: clip模型名
    :param clip_path: clip模型路径
    :param device: 设备
    """
    yolo = YOLO(yolo_path)
    clip_net, process = load_from_name(name=clip_head, device=device, download_root='/nas_data/WTY/cache')
    clip_net.load_state_dict(torch.load(clip_path))

    return yolo, clip_net, process

def get_img_path(file_path, prefix_path):
    """"
    获取图片路径
    :param file_path: 文件路径
    :param prefix_path: 图片前缀路径
    """

    img_path = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img = line.strip("\n").split(" ")[0]
            img_path.append(prefix_path + img)
    
    return img_path

def get_all_valid_words(img_ch_dir=r'data/'):
    """
    获取所有的有效字符
    :param img_ch_dir: 字符图片目录
    """



    valid_index_path = img_ch_dir + 'valid/index_img.txt'
    f = open(valid_index_path, mode='r', encoding='utf-8')
    data = f.readlines()
    f.close()
    valid_words = []
    for item in data:
        item = item.strip().split('\t')
        if len(item) == 2:
            item = item[1]
        else:
            item = ''
        if item not in valid_words:
            valid_words.append(item)
    return valid_words


def process_loop(img, yolo, clip_net, process, valid_tensors, id2word, device):
        """
        处理循环
        :param img: 图片路径
        :param yolo: yolo模型
        :param clip_net: clip模型
        :param process: 图片处理函数
        :param valid_tensors: 有效字符张量
        :param id2word: id2word
        :param device: 设备
        """
        
        clip_net.eval()
        # with suppress_stdout_stderr():
        results = yolo.predict([img], conf=0.5)
        r = results[0]
        img = Image.open(img)
        img = img.convert('RGB')
        size = img.size
        boxes = r.boxes.xywhn.cpu().numpy()

        if len(boxes) == 0:
            return ''

        boxes = convert_boxes(boxes, size)
        
        sb1, sb2, sb3, sb4 = regularization_sorting(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], alpha=45,
                                                    beta=30)
        sb1, sb2, sb3, sb4 = np.expand_dims(sb1, axis=1), np.expand_dims(sb2, axis=1), np.expand_dims(sb3, axis=1), np.expand_dims(sb4, axis=1)
        boxes = np.concatenate([sb1, sb2, sb3, sb4], axis=1)


        
        sim_results = []

        with torch.no_grad():
            for i, box in enumerate(boxes):
                x, y, w, h = box
                crop = img.crop((x, y, x + w, y + h))
                crop = process(crop)
                crop = crop.unsqueeze(0).to(device)
                img_tensor = clip_net.encode_image(crop)
                img_tensor = img_tensor / img_tensor.norm(dim=-1, keepdim=True)
                sim_mtrix = torch.nn.functional.cosine_similarity(valid_tensors.unsqueeze(0), img_tensor.unsqueeze(1), dim=-1)
                sim_result = torch.argmax(sim_mtrix, dim=-1).cpu().numpy().tolist()
                sim_results += sim_result



        text = ''
        for i, sim_result in enumerate(sim_results):
            text += id2word[sim_result]

        return text


def main():

    device = torch.device('cuda')
    yolo, clip_net, process = load_model_and_process('/nas_data/WTY/project/nlp_task1/runs/detect/train2/weights/best.pt',
                                                     'RN50',
                                                     '/nas_data/WTY/project/nlp_task1/clip_demo/model/RN50_epoch_7.pth',
                                                     device
                                                     )
    
    process = image_transform(224)

    img_path = get_img_path('/nas_data/WTY/dataset/visualC3/valid/label/tgt_valid.txt', '/nas_data/WTY/dataset/visualC3/valid/imgs/')


    all_valid_words = get_all_valid_words('/nas_data/WTY/dataset/visualC3/char/')
    with torch.no_grad():
        valid_token = clip.tokenize(all_valid_words, context_length=6).to(torch.device(device))
        valid_tensors = clip_net.encode_text(valid_token)
    id2word = {i: word.strip(' ') for i, word in enumerate(all_valid_words)}

    with open('result.txt', 'w') as f:
        for img in tqdm(img_path):
            text = process_loop(img, yolo, clip_net, process, valid_tensors ,id2word ,device)
            img = img.removeprefix('/nas_data/WTY/dataset/visualC3/valid/imgs/')
            f.write(img + "  " + text + "\n")


if __name__ == "__main__":
    main()