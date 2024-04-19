import torch
from PIL import Image
from tqdm import tqdm

from cn_clip import clip
from cn_clip.clip import load_from_name
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


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
            imge_path.append(img_ch_dir + mode + '/' + item[0].split('/')[-1])
            character.append(item[1])
        else:
            imge_path.append(img_ch_dir + mode + '/' + item[0].split('/')[-1])
            character.append('')
    return imge_path, character


class Img_Ch_Dataset(Dataset):
    def __init__(self, image_path, character, process):
        super(Img_Ch_Dataset, self).__init__()
        self.image_path = image_path
        self.character = character
        self.tokens = clip.tokenize(self.character)
        self.process = process

    def __getitem__(self, index):
        image = self.image_path[index]
        image = Image.open(image).convert('RGB')
        image = self.process(image)
        token = self.tokens[index]
        return image, token

    def __len__(self):
        return len(self.character)


def get_data_loader(process, batch_size=1, img_ch_dir=r'./data/', mode='train'):
    image_path, character = get_img_ch(img_ch_dir, mode=mode)
    data_set = Img_Ch_Dataset(image_path, character, process=process)
    if mode == 'train':
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4,
                                 drop_last=False)
    else:
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
        if item not in valid_words:
            valid_words.append(item)
    return valid_words


def get_valid_word_index_list(img_ch_dir):
    valid_imgs, valid_words = get_img_ch(img_ch_dir=img_ch_dir, mode='valid')
    all_valid_words = get_all_valid_words()
    valid_index_list = []
    for item in valid_words:
        valid_index_list.append(all_valid_words.index(item))
    return valid_index_list, all_valid_words


def valid_method(img_ch_dir, device, clip_net, process):
    valid_index_list, all_valid_words = get_valid_word_index_list(img_ch_dir=img_ch_dir)
    valid_data_loader = get_data_loader(process=process, batch_size=64, img_ch_dir=img_ch_dir, mode='valid')
    valid_token = clip.tokenize(all_valid_words).to(torch.device(device))
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


if __name__ == '__main__':
    # valid_words = get_all_valid_words()
    # valid_token = clip.tokenize(valid_words).to(torch.device('cuda:0'))
    # clip_net, process = load_from_name(name='RN50', device=torch.device('cuda:0'), download_root='./clip_model')
    # valid_tensors = clip_net.encode_text(valid_token)
    # print(valid_tensors.shape)
    _, _, vlid_index = get_valid_word_index_list(img_ch_dir=r'./data/')
    print(vlid_index)
