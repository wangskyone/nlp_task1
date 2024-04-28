import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.optim import lr_scheduler
from tqdm import tqdm

from cn_clip.clip import load_from_name
from img_ch_dataset import get_data_loader, valid_method, valid_method_K, valid_save_error_image, valid_top10_retrieval
import os
from utils import image_transform
from accelerate.utils import set_seed
from torch.nn import functional as F



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

# RN50 seed 3942374709
# vit seed 2316182608



class LinerProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinerProjection, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x

def train(args):
    print(args)

    # # 获取当前自动选择的随机数种子
    # current_seed = torch.initial_seed() % 2**32

    # # 打印本次运行的随机数种子
    # print(f"Random seed used: {current_seed}")


    set_seed(2316182608)

    accum_iter = 1 

    logger.info('start training!')
    device = torch.device(args.device)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        logger.info("Create model dir:{}".format(args.model_dir))
    clip_net, process = load_from_name(name=args.clip_img_head, device=device, download_root='/nas_data/WTY/cache')
    process = image_transform(224)
    optimizer = optim.AdamW(clip_net.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps,
                            weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    data_loader = get_data_loader(process=process, batch_size=args.batch_size, img_ch_dir=args.img_ch_dir,
                                  mode=args.mode)
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
                with torch.set_grad_enabled(args.mode == 'train'):
                    imgs_tensor = clip_net.encode_image(imgs)
                    token_tensor = clip_net.encode_text(tokens)
                    imgs_tensor = imgs_tensor / imgs_tensor.norm(dim=-1, keepdim=True)
                    token_tensor = token_tensor / token_tensor.norm(dim=-1, keepdim=True)
                    # loss = correct_clip_loss(image_features=imgs_tensor, text_features=token_tensor, logit_scale=logit_scale, device=device ,id=data[-1])
                    loss = clip_loss(img_tensor=imgs_tensor, ch_tensor=token_tensor, device=args.device,logit_scale=logit_scale)        
                    
                    loss = loss / accum_iter
                    loss.backward()

                    if ((idx + 1) % accum_iter == 0) or (idx + 1 == len(data_loader)):
                        optimizer.step()
                        optimizer.zero_grad()

                if batch_num % args.loss_step == 0:
                    logger.info('step:{},loss:{}'.format(str(batch_num), str(float(loss))))
                if batch_num % args.valid_step == 0:
                    clip_net.eval()
                    acc = valid_method(img_ch_dir=args.img_ch_dir, device=args.device, clip_net=clip_net,
                                       process=process)
                    if acc > best_acc:
                        best_acc = acc
                        logger.info('higher acc:{}, save model'.format(str(acc)))
                        torch.save(clip_net.state_dict(), args.model_dir + f'{args.clip_img_head}_epoch_{str(batch_num)}.pth')
                    logger.info('acc:{} in step:{},loss:{}'.format(str(acc), str(batch_num), str(float(loss))))

            scheduler.step()

def valid(args):
    logger.info('start valid!')
    device = torch.device(args.device)
    clip_net, process = load_from_name(name=args.clip_img_head, device=device, download_root='/nas_data/WTY/cache')
    process = image_transform(224)
    clip_net.load_state_dict(torch.load('/nas_data/WTY/project/nlp_task1/clip/model/RN50_epoch_40200.pth'))
    clip_net.eval()
    # acc1 = valid_top10_retrieval(img_ch_dir=args.img_ch_dir, device=args.device, clip_net=clip_net, process=process)
    # print(acc)
    valid_save_error_image(img_ch_dir=args.img_ch_dir, device=args.device, clip_net=clip_net, process=process)
    # acc_1, acc_5, acc_10, acc_100 = valid_method_K(img_ch_dir=args.img_ch_dir, device=args.device, clip_net=clip_net, process=process, K=10)
    # logger.info(f'acc@1:{acc_1}, acc@5:{acc_5}, acc@10:{acc_10}, acc@100:{acc_100}')

    # acc1 = valid_method(img_ch_dir=args.img_ch_dir, device=args.device, clip_net=clip_net, process=process,)
    # logger.info(f'acc@1:{acc1}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=r'./model_test/',
                        help='path to model')
    parser.add_argument('--img_ch_dir', type=str, default=r'/nas_data/WTY/dataset/visualC3/char/',
                        help='path to datasets')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test'],
                        help='train,valid or test')
    parser.add_argument('--batch_size', type=int, default=256, help='the size of a batch')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'])
    parser.add_argument('--learning_rate', type=float, default=2e-05)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.98))
    parser.add_argument('--eps', type=float, default=2e-05)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--valid_step', type=int, default=600)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--loss_step', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--clip_img_head', type=str, default='RN50',
                        choices=['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50'])
    args = parser.parse_args()
    train(args)
    # valid(args)

if __name__ == '__main__':
    main()
