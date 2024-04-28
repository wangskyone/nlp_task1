import datasets
from torch.utils.data import Dataset
import numpy as np

class CorrectDataset(Dataset):

    def __init__(self, src_file_path, trg_file_path, mode, tokenizer):
        super(CorrectDataset, self).__init__()

        self.src_dataset = datasets.load_dataset('text', data_files={mode: src_file_path}, cache_dir='/nas_data/WTY/cache')
        self.trg_dataset = datasets.load_dataset('text', data_files={mode: trg_file_path}, cache_dir='/nas_data/WTY/cache')
        self.tokenizer = tokenizer
        def src_process(example):
            new_example = {'src_ids': []}
            for i in range(len(example['text'])):
                line = example['text'][i]
                text = line.split(' ')[-1].strip('\n')
                text = text.replace('U','<U>').replace('X','<X>')
                inputs = tokenizer.encode(text, return_tensors="pt", padding="max_length", max_length=100, truncation=True).squeeze(0)
                new_example['src_ids'].append(inputs)

            return new_example
        
        def trg_process(example):
            
            new_example = {'trg_ids': []}
            for i in range(len(example['text'])):
                line = example['text'][i]
                text = line.split(' ')[-1].strip('\n')
                text = text.replace('U','<U>').replace('X','<X>')
                inputs = tokenizer.encode(text, return_tensors="pt", padding="max_length", max_length=100, truncation=True).squeeze(0)
                new_example['trg_ids'].append(inputs)

            return new_example
        
        self.src_dataset = self.src_dataset.map(src_process, remove_columns=['text'], batched=True)
        self.trg_dataset = self.trg_dataset.map(trg_process, remove_columns=['text'], batched=True)

        self.dataset = datasets.concatenate_datasets([self.src_dataset[mode], self.trg_dataset[mode]], axis=1)
        self.dataset.set_format(type='torch', columns=['src_ids', 'trg_ids'])

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        return self.dataset[idx]['src_ids'], self.dataset[idx]['trg_ids']
    


if __name__ == "__main__":
    from transformers import BertTokenizer, BartForConditionalGeneration
    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-large-chinese", cache_dir="/nas_data/WTY/cache")
    tokenizer.add_special_tokens({"additional_special_tokens":['<U>','<X>']})
    dataset = CorrectDataset('/nas_data/WTY/dataset/visualC3/train/label/src_train.txt',
                             '/nas_data/WTY/dataset/visualC3/train/label/tgt_train.txt',
                             'train', tokenizer)
    
    lens = []

    for i in range(len(dataset)):

        lens.append(len(dataset[i]['src_ids']))

    print(np.mean(lens), np.max(lens), np.min(lens))