import torch
from PIL import Image
import torch.utils
from tqdm import tqdm

from cn_clip import clip
from cn_clip.clip import load_from_name
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score
import numpy as np
import random
from utils import Q2B

def get_img_ch(img_ch_dir=r'./data/', mode='train'):
    """
    :param img_ch_dir: data directory
    :param mode: train,valid or test
    :return:the list of image path and the corresponding character
    """
    index_file_path = img_ch_dir + mode + '/' + 'index_img.txt'
    f = open(index_file_path, mode='r', encoding='utf-8')
    data = f.readlines()
    f.close()
    imge_path = []
    character = []
    for item in data:
        item = item.strip().split('\t')
        if len(item) == 2:
            item[1] = ''.join([Q2B(ch) for ch in item[1]])
            t = item[1].strip()
            if t == "X":
                continue
            character.append(item[1].strip())
            imge_path.append(img_ch_dir + mode + '/' + item[0].split('/')[-1])
        else:
            character.append('')
            imge_path.append(img_ch_dir + mode + '/' + item[0].split('/')[-1])
    return imge_path, character


class Img_Ch_Dataset(Dataset):
    def __init__(self, image_path, character, process):
        super(Img_Ch_Dataset, self).__init__()
        self.image_path = image_path
        self.character = character
        self.tokens = clip.tokenize(self.character, context_length=6)
        self.process = process

    def __getitem__(self, index):
        image = self.image_path[index]
        image = Image.open(image).convert('RGB')
        image = self.process(image)
        token = self.tokens[index]
        return image, token

    def __len__(self):
        return len(self.character)
    

class Img_Ch_Dataset_Train(Dataset):
    def __init__(self, image_path, character, process):
        super(Img_Ch_Dataset_Train, self).__init__()
        self.image_path = image_path
        self.character = character


        dic = {}
        t = 0
        for i in range(len(character)):
            if character[i] not in dic:
                dic[character[i]] = t
                t += 1
        self.label = [dic[item] for item in character]
        self.tokens = clip.tokenize(self.character, context_length=6)
        self.process = process

    def __getitem__(self, index):
        image = self.image_path[index]
        image = Image.open(image).convert('RGB')
        image = self.process(image)
        token = self.tokens[index]
        return image, token, torch.tensor(self.label[index])

    def __len__(self):
        return len(self.character)


class Img_Ch_Dataset_U(Dataset):
    def __init__(self, image_path, character, process):
        super(Img_Ch_Dataset_U, self).__init__()

        dic = {}
        for i in range(len(character)):
            if character[i] not in dic:
                dic[character[i]] = []
            dic[character[i]].append(image_path[i])

        del dic['X']

        self.image_path = []
        self.character = []

        for key in dic.keys():
            self.image_path += dic[key]
            self.character += [key] * len(dic[key])
        
        self.tokens = clip.tokenize(self.character, context_length=6)
        self.process = process
        
    def __getitem__(self, index):
        image = self.image_path[index]
        image = Image.open(image).convert('RGB')
        image = self.process(image)
        token = self.tokens[index]
        return image, token

    def __len__(self):
        return len(self.image_path)
    



def get_data_loader(process, batch_size=1, img_ch_dir=r'./data/', mode='train'):
    image_path, character = get_img_ch(img_ch_dir, mode=mode)
    if mode == 'train':
        data_set = Img_Ch_Dataset(image_path, character, process=process)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4,
                                 drop_last=False)
    else:
        data_set = Img_Ch_Dataset(image_path, character, process=process)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=4,
                                 drop_last=False)
    return data_loader



def get_all_valid_words(valid_index_path=r'data/valid/index_img.txt'):
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
        item = ''.join([Q2B(ch) for ch in item])
        item = item.strip()
        if item == 'X':
            continue
        if item not in valid_words:
            valid_words.append(item)
    return valid_words


def get_valid_word_index_list(img_ch_dir):
    valid_imgs, valid_words = get_img_ch(img_ch_dir=img_ch_dir, mode='valid')
    all_valid_words = get_all_valid_words(img_ch_dir + 'valid/index_img.txt')
    valid_index_list = []
    for item in valid_words:
        valid_index_list.append(all_valid_words.index(item))
    return valid_index_list, all_valid_words

def get_valid_img_word_index_list(img_ch_dir):
    valid_imgs, valid_words = get_img_ch(img_ch_dir=img_ch_dir, mode='valid')
    all_valid_words = get_all_valid_words(img_ch_dir + 'valid/index_img.txt')
    valid_index_list = []
    for item in valid_words:
        valid_index_list.append(all_valid_words.index(item))
    return valid_imgs, valid_index_list, all_valid_words


def valid_method(img_ch_dir, device, clip_net, process):
    valid_index_list, all_valid_words = get_valid_word_index_list(img_ch_dir=img_ch_dir)
    valid_data_loader = get_data_loader(process=process, batch_size=128, img_ch_dir=img_ch_dir, mode='valid')
    valid_token = clip.tokenize(all_valid_words, context_length=6).to(torch.device(device))
    valid_tensors = clip_net.encode_text(valid_token)
    sim_results = []
    for idx, data in enumerate(tqdm(valid_data_loader)):
        imgs = data[0].to(device)
        with torch.no_grad():
            imgs_tensor = clip_net.encode_image(imgs)
            imgs_tensor = imgs_tensor / imgs_tensor.norm(dim=-1, keepdim=True)
            sim_mtrix = F.cosine_similarity(valid_tensors.unsqueeze(0), imgs_tensor.unsqueeze(1), dim=-1)
            sim_result = torch.argmax(sim_mtrix, dim=-1).cpu().numpy().tolist()
            sim_results += sim_result
    acc = accuracy_score(valid_index_list, sim_results)
    return acc


def valid_method_K(img_ch_dir, device, clip_net, process, K):
    valid_index_list, all_valid_words = get_valid_word_index_list(img_ch_dir=img_ch_dir)
    valid_data_loader = get_data_loader(process=process, batch_size=128, img_ch_dir=img_ch_dir, mode='valid')
    valid_token = clip.tokenize(all_valid_words, context_length=6).to(torch.device(device))
    valid_tensors = clip_net.encode_text(valid_token)
    sim_results = []
    for idx, data in enumerate(tqdm(valid_data_loader)):
        imgs = data[0].to(device)
        with torch.no_grad():
            imgs_tensor = clip_net.encode_image(imgs)
            imgs_tensor = imgs_tensor / imgs_tensor.norm(dim=-1, keepdim=True)
            sim_mtrix = F.cosine_similarity(valid_tensors.unsqueeze(0), imgs_tensor.unsqueeze(1), dim=-1)
            sim_result = sim_mtrix.cpu().numpy().tolist()
            sim_results += sim_result
    
    valid_index_list, sim_results = np.array(valid_index_list), np.array(sim_results)
    
    acc_1 = accuracy_score(valid_index_list, sim_results.argmax(axis=1))
    acc_5 = top_k_accuracy_score(y_true=valid_index_list, y_score=sim_results, k=5)
    acc_10 = top_k_accuracy_score(y_true=valid_index_list, y_score=sim_results, k=10)
    acc_100 = top_k_accuracy_score(y_true=valid_index_list, y_score=sim_results, k=100)
    return acc_1, acc_5, acc_10, acc_100


def valid_save_error_image(img_ch_dir, device, clip_net, process):
    valid_imgpath_list, valid_index_list, all_valid_words = get_valid_img_word_index_list(img_ch_dir=img_ch_dir)
    valid_data_loader = get_data_loader(process=process, batch_size=128, img_ch_dir=img_ch_dir, mode='valid')
    valid_token = clip.tokenize(all_valid_words, context_length=6).to(torch.device(device))
    valid_tensors = clip_net.encode_text(valid_token)

    sim_results = []

    for idx, data in enumerate(tqdm(valid_data_loader)):
        imgs = data[0].to(device)
        with torch.no_grad():
            imgs_tensor = clip_net.encode_image(imgs)
            imgs_tensor = imgs_tensor / imgs_tensor.norm(dim=-1, keepdim=True)
            sim_mtrix = F.cosine_similarity(valid_tensors.unsqueeze(0), imgs_tensor.unsqueeze(1), dim=-1)
            sim_result = torch.argmax(sim_mtrix, dim=-1).cpu().numpy().tolist()
            sim_results += sim_result

    err_index = []

    for i in tqdm(range(len(valid_index_list))):
        if valid_index_list[i] != sim_results[i]:
            err_index.append(i)

    with open('error_index.txt', 'w') as f:
        for i in err_index:
            f.write(valid_imgpath_list[i] + '  ' + all_valid_words[valid_index_list[i]] + '  ' + all_valid_words[sim_results[i]] + '\n')



def valid_top10_retrieval(img_ch_dir, device, clip_net, process, p=0.05):
    valid_imgpath_list, valid_index_list, all_valid_words = get_valid_img_word_index_list(img_ch_dir=img_ch_dir)
    valid_data_loader = get_data_loader(process=process, batch_size=128, img_ch_dir=img_ch_dir, mode='valid')
    valid_token = clip.tokenize(all_valid_words, context_length=6).to(torch.device(device))
    valid_tensors = clip_net.encode_text(valid_token)

    sim_results = []
    for idx, data in enumerate(tqdm(valid_data_loader)):
        imgs = data[0].to(device)
        with torch.no_grad():
            imgs_tensor = clip_net.encode_image(imgs)
            imgs_tensor = imgs_tensor / imgs_tensor.norm(dim=-1, keepdim=True)
            sim_mtrix = F.cosine_similarity(valid_tensors.unsqueeze(0), imgs_tensor.unsqueeze(1), dim=-1)
            sim_result = torch.topk(sim_mtrix, k=10, dim=1).indices.cpu().numpy().tolist()
            sim_results += sim_result

    
    image_path, char = get_img_ch('/nas_data/WTY/dataset/visualC3/char/', mode='train')

    dic = {}

    for i in range(len(image_path)):
        char[i] = char[i].strip()
        if char[i] not in dic:
            dic[char[i]] = [image_path[i]]
        else:
            dic[char[i]].append(image_path[i])

    valid_data_loader = get_data_loader(process=process, batch_size=1, img_ch_dir=img_ch_dir, mode='valid')


    dic_tensors = torch.load('dic_tensor.pt')
    dic_sims = torch.load('dic_sims.pt')

    with torch.no_grad():
        for idx, data in enumerate(tqdm(valid_data_loader)):
            imgs = data[0].to(device)
            imgs_tensor = clip_net.encode_image(imgs)
            imgs_tensor = imgs_tensor / imgs_tensor.norm(dim=-1, keepdim=True)
            all_sims = []
            if all_valid_words[sim_results[idx][0]] == "U":
                sim_results[idx] = all_valid_words.index('U')
                continue
            
            for t in sim_results[idx]:
                word = all_valid_words[t].strip()
                if word not in dic:
                    continue

                sims = F.cosine_similarity(dic_tensors[word].to(device), imgs_tensor, dim=-1).cpu().numpy()

                all_sims.append(sims.mean())

            all_sims = np.array(all_sims)


            index = all_sims.argmax()
            # if all_sims.max() < dic_sims[all_valid_words[index]] - p:
            #     sim_results[idx] = all_valid_words.index('X')
            # else:
            sim_results[idx] = sim_results[idx][index]
            # if all_sims.max() < conf:
            #     sim_results[idx] = all_valid_words.index('X')
            # else:
            #     sim_results[idx] = sim_results[idx][all_sims.argmax()]

    acc = accuracy_score(valid_index_list, sim_results)

    err_index = []

    for i in tqdm(range(len(valid_index_list))):
        if valid_index_list[i] != sim_results[i]:
            err_index.append(i)

    with open('error_index.txt', 'w') as f:
        for i in err_index:
            f.write(valid_imgpath_list[i] + '  ' + all_valid_words[valid_index_list[i]] + '  ' + all_valid_words[sim_results[i]] + '\n')


    return acc


if __name__ == '__main__':
    # valid_words = get_all_valid_words()
    # valid_token = clip.tokenize(valid_words).to(torch.device('cuda:0'))
    # clip_net, process = load_from_name(name='RN50', device=torch.device('cuda:0'), download_root='./clip_model')
    # valid_tensors = clip_net.encode_text(valid_token)
    # print(valid_tensors.shape)
    _, _, vlid_index = get_valid_word_index_list(img_ch_dir=r'./data/')
    print(vlid_index)
