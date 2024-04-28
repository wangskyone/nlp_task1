import torch.utils
from transformers import BartForConditionalGeneration, BertTokenizer, TrainingArguments
from transformers.trainer import Trainer, TrainerCallback
from trainer import CorrectTrainer
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score
from correct_dataset import CorrectDataset

# os.environ["WANDB_API_KEY"] = "4083f56e4bba6c43fad3916f594c830bd98d1ac8"
# os.environ["WANDB_MODE"] = "online"

def compute_metrics(eval_preds):
    num_predict = 0
    num_correct = 0
    k = 1
    for predict, label in zip(eval_preds.predictions, eval_preds.label_ids):
        num_predict += 1

        if k > 0:
            print("predict:", predict)
            print("label:", label)
            print(label[np.where(label == 101)[0].item() + 1:np.where(label == 102)[0].item()])
            print(predict[np.where(predict == 101)[0].item()+1:np.where(predict == 102)[0][-1].item()])
            k -= 1
        if np.array_equal(label[np.where(label == 101)[0].item() + 1:np.where(label == 102)[0].item()],
                          predict[np.where(predict == 101)[0].item()+1:np.where(predict == 102)[0][-1].item()]):
            num_correct += 1

    return {'accuracy': num_correct / num_predict}

def main():

    # wandb.login()
    # wandb.init(project="DSI_GPT", name='mscoco10K')
    tokenizer = BertTokenizer.from_pretrained("/nas_data/WTY/cache/models--fnlp--bart-large-chinese/snapshots/75cdf21ffc77809dd8cd5fd52d552f3bb35eafc3/")
    model = BartForConditionalGeneration.from_pretrained("/nas_data/WTY/cache/models--fnlp--bart-large-chinese/snapshots/75cdf21ffc77809dd8cd5fd52d552f3bb35eafc3/")

    tokenizer.add_special_tokens({"additional_special_tokens":['<U>','<X>']})
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = CorrectDataset('/nas_data/WTY/dataset/visualC3/train/label/src_train.txt',
                             '/nas_data/WTY/dataset/visualC3/train/label/tgt_train.txt',
                             'train', tokenizer)
    
    eval_dataset = CorrectDataset('/nas_data/WTY/dataset/visualC3/valid/label/src_valid.txt',
                             '/nas_data/WTY/dataset/visualC3/valid/label/tgt_valid.txt',
                             'valid', tokenizer)
    
    eval_dataset, _ = torch.utils.data.random_split(eval_dataset, [0.1, 0.9])

    # print(train_dataset)
    # print(eval_dataset)

    # exit()
    # tokenizer.pad_token = tokenizer.eos_token
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=IndexingCollator(tokenizer, padding='longest'))
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    # model = CoCa(embed_dim=768, vision_cfg=vision_cfg, text_cfg=text_cfg, multimodal_cfg=multimodal_cfg)
    # weight_dict = torch.load("/nas_data/WTY/cache/CoCa/open_clip_pytorch_model.bin")
    # model.load_state_dict({k.replace("module.", ""): v for k, v in weight_dict.items()},strict=False)
    # for name, param in model.named_parameters():
    #     if "text_decoder" not in name:
    #         param.requires_grad = False
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    # exit()    

    # model = torch.compile(model)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        evaluation_strategy='steps',
        eval_steps=300,
        max_steps=10000,
        dataloader_drop_last=False,  # necessary
        report_to='none',
        logging_steps=300,
        save_strategy='no',
        save_steps=300,
        # fp16=True,  # gives 0/nan loss at some point during training, seems this is a transformers bug.
        dataloader_num_workers=8,
        # gradient_accumulation_steps=2
    )

    trainer = CorrectTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        # callbacks=[QueryEvalCallback(eval_dataset, wandb, restrict_decode_vocab, training_args, tokenizer)],
    )

    trainer.train(
    )

if __name__ == "__main__":
    main()
