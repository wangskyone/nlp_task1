from PIL import Image
from cn_clip import clip
from torch.utils.data import Dataset, DataLoader


def get_img_ch(img_ch_dir=r'./data/', mode='train'):
    """
    :param img_ch_dir: data directory
    :param mode: train,valid or test
    :return:the list of image path and the corresponding character
    """
    index_file_path = img_ch_dir + mode + '/' + 'img_ch.txt'
    f = open(index_file_path, mode='r', encoding='utf-8')
    data = f.readlines()
    f.close()
    imge_path = []
    character = []
    for item in data:
        item = item.strip().split('\t')
        imge_path.append(img_ch_dir + mode + '/' + item[0])
        character.append(item[1])
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
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
    else:
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
    return data_loader


if __name__ == '__main__':
    data_loader = get_data_loader()
    for img, token in data_loader:
        print(img)
        print(token)
