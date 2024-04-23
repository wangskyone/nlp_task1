import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
from cn_clip import clip
from cn_clip.clip import load_from_name
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
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


class Img_Ch_Dataset_Common(Dataset):
    def __init__(self, image_path, character, text_encoder_path, process):
        super(Img_Ch_Dataset_Common, self).__init__()
        self.image_path = image_path
        self.character = character
        self.process = process
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)

    def __getitem__(self, index):
        image = self.image_path[index]
        image = Image.open(image).convert('RGB')
        image = self.process(image)
        ch = self.character[index]
        ch_token = self.tokenizer(ch, max_length=6, padding='max_length',
                                  return_tensors='pt')
        bert_input = [ch_token['input_ids'].squeeze(0), ch_token['token_type_ids'].squeeze(0),
                      ch_token['attention_mask'].squeeze(0)]
        return image, bert_input

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


def get_data_loader_common(process, text_encoder_path=r'chinese-bert-base', batch_size=1, img_ch_dir=r'./data/',
                           mode='train'):
    image_path, character = get_img_ch(img_ch_dir, mode=mode)
    data_set = Img_Ch_Dataset_Common(image_path, character, text_encoder_path=text_encoder_path, process=process)
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
    valid_index_list,all_valid_words = get_valid_word_index_list(img_ch_dir=img_ch_dir)
    valid_data_loader = get_data_loader(process=process, batch_size=64, img_ch_dir=img_ch_dir, mode='valid')
    valid_token = clip.tokenize(all_valid_words, context_length=6).to(device)
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


def valid_method_common(img_ch_dir, device, clip_net, process, valid_token):
    valid_index_list = get_valid_word_index_list(img_ch_dir=img_ch_dir)
    valid_data_loader = get_data_loader(process=process, batch_size=64, img_ch_dir=img_ch_dir, mode='valid')
    valid_tensors = clip_net(image_input=None, input_idx=valid_token.input_ids,
                             attention_mask=valid_token.attention_mask, token_type_ids=valid_token.token_type_ids,
                             mode='word')
    sim_results = []
    for idx, data in enumerate(tqdm(valid_data_loader)):
        imgs = data[0].to(device)
        with torch.no_grad():
            imgs_tensor = clip_net(image_input=imgs, input_idx=None, attention_mask=None, token_type_ids=None,
                                   mode='image')
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
    a = ['我', '你']
    tokenizer = AutoTokenizer.from_pretrained(r'chinese-bert-base')
    a = tokenizer(a, max_length=6, padding='max_length',
                  return_tensors='pt')
    bert = AutoModel.from_pretrained(r'chinese-bert-base')
    a = bert(input_ids=a.input_ids, attention_mask=a.attention_mask, token_type_ids=a.token_type_ids).last_hidden_state[
        :, 0]
    print(a)
    print(a.shape)
