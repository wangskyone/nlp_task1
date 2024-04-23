import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer

from cn_clip import clip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.optim import lr_scheduler
from tqdm import tqdm
from model_common import model_common
from torchvision import transforms

from cn_clip.clip import load_from_name
from img_ch_dataset import get_data_loader, valid_method, get_data_loader_common, get_all_valid_words, \
    get_valid_word_index_list, valid_method_common


def clip_loss(img_tensor, ch_tensor, logit_scale, device):
    logit_scale = logit_scale.exp()
    logits_per_image = logit_scale * img_tensor @ ch_tensor.t()
    logits_per_text = logits_per_image.t()
    ground_truth = torch.arange(len(img_tensor), dtype=torch.long, device=device)
    loss_fct = nn.CrossEntropyLoss()
    loss_img = loss_fct(logits_per_image, ground_truth)
    loss_token = loss_fct(logits_per_text, ground_truth)
    loss = (loss_token + loss_img) / 2
    return loss


def train(args):
    print(args)
    logger.info('start training!')
    device = torch.device(args.device)
    if args.model_mode == 'clip':
        clip_net, process = load_from_name(name=args.clip_img_head, device=device, download_root='./clip_model')
        data_loader = get_data_loader(process=process, batch_size=args.batch_size, img_ch_dir=args.img_ch_dir,
                                      mode=args.mode)
        optimizer = optim.AdamW(clip_net.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps,
                                weight_decay=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        best_acc = 0
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        batch_num = 0
        for i in range(args.epoch):
            with torch.cuda.amp.autocast(enabled=True):
                for idx, data in tqdm(enumerate(tqdm(data_loader))):
                    clip_net.train()
                    imgs = data[0].to(device)
                    tokens = data[1].to(device)
                    batch_num += 1
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(args.mode == 'train'):
                        imgs_tensor = clip_net.encode_image(imgs)
                        token_tensor = clip_net.encode_text(tokens)
                        imgs_tensor = imgs_tensor / imgs_tensor.norm(dim=-1, keepdim=True)
                        token_tensor = token_tensor / token_tensor.norm(dim=-1, keepdim=True)
                        loss = clip_loss(img_tensor=imgs_tensor, ch_tensor=token_tensor, device=args.device,
                                         logit_scale=logit_scale)
                        loss.backward()
                        optimizer.step()
                    if batch_num % args.loss_step == 0:
                        logger.info('step:{},loss:{}'.format(str(batch_num), str(float(loss))))
                    if batch_num % args.valid_step == 0:
                        clip_net.eval()
                        acc = valid_method(img_ch_dir=args.img_ch_dir, device=args.device, clip_net=clip_net,
                                           process=process)
                        if acc > best_acc:
                            best_acc = acc
                            logger.info('higher acc:{}, save model'.format(str(acc)))
                            torch.save(clip_net.state_dict(), args.model_dir + f'{args.clip_img_head}.pth')
                        logger.info('acc:{} in step:{},loss:{}'.format(str(acc), str(batch_num), str(float(loss))))

            scheduler.step()
    else:
        all_valid_words = get_all_valid_words()
        clip_net = model_common(text_encoder_path=args.text_encoder_path).to(device)
        norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        process = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ]
        )
        data_loader = get_data_loader_common(process=process, batch_size=args.batch_size, img_ch_dir=args.img_ch_dir,
                                             mode=args.mode)
        optimizer = optim.AdamW(clip_net.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps,
                                weight_decay=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        best_acc = 0
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        batch_num = 0
        tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path)
        valid_token = tokenizer(all_valid_words, max_length=6, padding='max_length',
                                return_tensors='pt').to(device)
        for i in range(args.epoch):
            with torch.cuda.amp.autocast(enabled=True):
                for idx, data in tqdm(enumerate(tqdm(data_loader))):
                    clip_net.train()
                    imgs = data[0].to(device)
                    input_idx = data[1][1].to(device)
                    attention_mask = data[1][1].to(device)
                    token_ype_ids = data[1][2].to(device)
                    batch_num += 1
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(args.mode == 'train'):
                        imgs_tensor, token_tensor = clip_net.forward(image_input=imgs, input_idx=input_idx,
                                                                     attention_mask=attention_mask,
                                                                     token_type_ids=token_ype_ids)
                        imgs_tensor = imgs_tensor / imgs_tensor.norm(dim=-1, keepdim=True)
                        token_tensor = token_tensor / token_tensor.norm(dim=-1, keepdim=True)
                        loss = clip_loss(img_tensor=imgs_tensor, ch_tensor=token_tensor, device=args.device,
                                         logit_scale=logit_scale)
                        loss.backward()
                        optimizer.step()
                    if batch_num % args.loss_step == 0:
                        logger.info('step:{},loss:{}'.format(str(batch_num), str(float(loss))))
                    if batch_num % args.valid_step == 0:
                        clip_net.eval()
                        acc = valid_method_common(img_ch_dir=args.img_ch_dir, device=args.device, clip_net=clip_net,
                                                  process=process, valid_token=valid_token)
                        if acc > best_acc:
                            best_acc = acc
                            logger.info('higher acc:{}, save model'.format(str(acc)))
                            torch.save(clip_net.state_dict(), args.model_dir + f'{args.clip_img_head}_common.pth')
                        logger.info('acc:{} in step:{},loss:{}'.format(str(acc), str(batch_num), str(float(loss))))

            scheduler.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=r'./model/',
                        help='path to model')
    parser.add_argument('--text_encoder_path', type=str, default=r'chinese-bert-base',
                        help='path to bert or roberta')
    parser.add_argument('--img_ch_dir', type=str, default=r'./data/',
                        help='path to datasets')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test'],
                        help='train,valid or test')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of a batch')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'])
    parser.add_argument('--learning_rate', type=float, default=1e-06)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.98))
    parser.add_argument('--eps', type=float, default=1e-06)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--valid_step', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--loss_step', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--clip_img_head', type=str, default='RN50',
                        choices=['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50'])
    parser.add_argument('--model_mode', type=str, default='common',
                        choices=['clip', 'common'])
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
