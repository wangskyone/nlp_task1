from cn_clip.clip import load_from_name
from img_ch_dataset import get_img_ch, get_valid_img_word_index_list
import os
from utils import image_transform
import torch
from PIL import Image
from cn_clip import clip
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# /nas_data/WTY/dataset/visualC3/char/valid/1000_0001_6.jpg
# /nas_data/WTY/dataset/visualC3/char/valid/1163_0006_5.jpg
# /nas_data/WTY/dataset/visualC3/char/valid/138_0005_3.jpg
# /nas_data/WTY/dataset/visualC3/char/valid/1564_0005_34.jpg
# /nas_data/WTY/dataset/visualC3/char/valid/167_0009_4.jpg
# /nas_data/WTY/dataset/visualC3/char/valid/958_0006_25.jpg

img1 = "/nas_data/WTY/dataset/visualC3/char/valid/1564_0005_34.jpg"


valid_imgpath_list, valid_index_list, all_valid_words = get_valid_img_word_index_list(img_ch_dir='/nas_data/WTY/dataset/visualC3/char/')


image_path, char = get_img_ch('/nas_data/WTY/dataset/visualC3/char/', mode='train')

dic_train = {}
dic_valid = {}

for i in range(len(image_path)):
    char[i] = char[i].strip()
    if char[i] not in dic_train:
        dic_train[char[i]] = [image_path[i]]
    else:
        dic_train[char[i]].append(image_path[i])


def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e: #转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)

dic_train_new = {}

for key, value in dic_train.items():
    key_new = ''.join([Q2B(ch) for ch in key])
    if key_new not in dic_train_new:
        dic_train_new[key_new] = value
    else:
        dic_train_new[key_new] += value


print(dic_train_new["…"][:5])

# import os
# import shutil
# from glob import glob
 
# def mycopyfile(srcfile,dstpath):                       # 复制函数
#     if not os.path.isfile(srcfile):
#         print ("%s not exist!"%(srcfile))
#     else:
#         fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
#         if not os.path.exists(dstpath):
#             os.makedirs(dstpath)                       # 创建路径
#         shutil.copy(srcfile, dstpath + fname)          # 复制文件
#         print ("copy %s -> %s"%(srcfile, dstpath + fname))
 

# dst_dir = '/nas_data/WTY/dataset/U_copy/'                                    # 目的路径记得加斜杠
# src_file_list = dic_train['U']                                               
# for srcfile in src_file_list:
#     mycopyfile(srcfile, dst_dir)     



# dic_sims = torch.load('dic_sims.pt')




# print(dic_sims['有'])

# dic_tensors = torch.load('dic_tensor.pt')

# print(dic_tensors['有'])


# with open("../asset/3500.txt", "r", encoding='utf-8-sig') as f:
#     text = f.read().strip()

#     for ch in text:
#         if ch not in dic_train:
#             dic_train[ch] = []

# image_path, char = get_img_ch('/nas_data/WTY/dataset/visualC3/char/', mode='valid')

# for i in range(len(image_path)):
#     char[i] = char[i].strip()
#     if char[i] not in dic_valid:
#         dic_valid[char[i]] = [image_path[i]]
#     else:
#         dic_valid[char[i]].append(image_path[i])

# print(len(dic_train))
# print(len(dic_valid))
# # print(dic_train.keys() - dic_valid.keys())
# print(dic_valid.keys() - dic_train.keys())

# all_path = dic_train['有'][:10]
# device = torch.device('cuda')
# clip_net, process = load_from_name(name='RN50', device=device, download_root='/nas_data/WTY/cache')
# clip_net.load_state_dict(torch.load('/nas_data/WTY/project/nlp_task1/clip/model/RN50_epoch_40200.pth'))
# clip_net.eval()
# process = image_transform(224)


# dic_tensors = {}
# dic_sims = {}
# with torch.no_grad():
#     for key, value in tqdm(dic_train.items()):

#         for t in value[:5]:
#             img = Image.open(t).convert('RGB')
#             img = process(img)
#             img_tensor = clip_net.encode_image(img.unsqueeze(0).to(device))
#             img_tensor = img_tensor / img_tensor.norm(dim=-1, keepdim=True)
#             if key not in dic_tensors:
#                 dic_tensors[key] = [img_tensor]
#             else:
#                 dic_tensors[key].append(img_tensor)


#         while len(dic_tensors[key]) < 5:
#             dic_tensors[key].append(dic_tensors[key][0])

#         mask = (torch.ones(5, 5) - torch.eye(5)).to(device)
#         img_tensor = torch.cat(dic_tensors[key], dim=0)
#         sim = img_tensor @ img_tensor.T
#         sim = sim * mask
#         dic_sims[key] = sim.sum() / mask.sum()
#         dic_tensors[key] = img_tensor.cpu()


# torch.save(dic_tensors, "dic_tensor.pt")
# torch.save(dic_sims, "dic_sims.pt")

# img1 = Image.open(img1).convert('RGB')
# img1 = process(img1)
# img1_tensor = clip_net.encode_image(img1.unsqueeze(0).to(device))
# img1_tensor = img1_tensor / img1_tensor.norm(dim=-1, keepdim=True)


# valid_token = clip.tokenize(all_valid_words, context_length=6).to(torch.device(device))
# valid_tensors = clip_net.encode_text(valid_token)


# sim_mtrix = F.cosine_similarity(valid_tensors.unsqueeze(0), img1_tensor.unsqueeze(1), dim=-1)
# top10 = torch.topk(sim_mtrix, 10, dim=1).indices.squeeze(0).cpu().numpy().tolist()

# for t in top10:
#     word = all_valid_words[t].strip()
#     print(word)
#     all_path = dic_train[word][:100]

#     while len(all_path) < 100:
#         all_path += dic_train[word][:100 - len(all_path)]

#     sims = []
#     for img in all_path:
#         img = Image.open(img).convert('RGB')
#         img = process(img)
#         img_tensor = clip_net.encode_image(img.unsqueeze(0).to(device))
#         img_tensor = img_tensor / img_tensor.norm(dim=-1, keepdim=True)
#         sim = (img1_tensor @ img_tensor.T).item()
#         sims.append(sim)
    
#     # print(sims)
#     print(np.array(sims).mean())