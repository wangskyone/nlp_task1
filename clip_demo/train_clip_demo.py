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
import os
import numpy as np
from cn_clip.clip import load_from_name
from img_ch_dataset import get_data_loader, get_all_valid_words, valid_method
from utils import image_transform

def clip_loss(image_features, text_features, logit_scale ,device):
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    ground_truth = torch.arange(len(image_features), dtype=torch.long, device=device)
    loss_fct = nn.CrossEntropyLoss()
    loss_img = loss_fct(logits_per_image, ground_truth)
    loss_token = loss_fct(logits_per_text, ground_truth)
    loss = (loss_token + loss_img) / 2
    return loss

def correct_clip_loss(image_features, text_features, logit_scale ,accelerator, id):
    b = id.shape[0]
    gt = torch.eq(id.repeat(b, 1), id.repeat(b, 1).t()).float().to(accelerator.device)
    loss_fct = nn.CrossEntropyLoss()

    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    loss = (
        loss_fct(logits_per_image, gt)
        + loss_fct(logits_per_text, gt)
    ) / 2
    return loss


def train(args):
    print(args)
    # set_seed(3407)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        logger.info("Create model dir:{}".format(args.model_dir))
    logger.info('start training!')
    device = torch.device(args.device)
    clip_net, process = load_from_name(name=args.clip_img_head, device=device, download_root='/nas_data/WTY/cache')
    process = image_transform(224)
    optimizer = optim.AdamW(clip_net.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps,
                            weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    data_loader = get_data_loader(process=process, batch_size=args.batch_size, img_ch_dir=args.img_ch_dir,
                                  mode=args.mode)

    best_acc = 0
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    for i in range(args.epoch):
        with torch.cuda.amp.autocast(enabled=True):
            clip_net.train()
            for idx, data in tqdm(enumerate(tqdm(data_loader))):
                clip_net.train()
                imgs = data[0].to(device)
                tokens = data[1].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(args.mode == 'train'):
                    imgs_tensor = clip_net.encode_image(imgs)
                    token_tensor = clip_net.encode_text(tokens)
                    imgs_tensor = imgs_tensor / imgs_tensor.norm(dim=-1, keepdim=True)
                    token_tensor = token_tensor / token_tensor.norm(dim=-1, keepdim=True)
                    # loss = correct_clip_loss(image_features=image_features, text_features=text_features, logit_scale=logit_scale ,accelerator=accelerator, id=data[-1])
                    loss = clip_loss(image_features=imgs_tensor, text_features=token_tensor, logit_scale=logit_scale, device=device)
                    loss.backward()
                    optimizer.step()

                if (i * len(data_loader) + idx + 1) % 100 == 0:
                    logger.info('loss:{} in step:{}'.format(str(float(loss)), str(i * len(data_loader) + idx + 1)))

                if (i * len(data_loader) + idx + 1) % args.valid_step == 0:
                    clip_net.eval()
                    acc = valid_method(img_ch_dir=args.img_ch_dir, clip_net=clip_net,
                                        process=process, device=device)
                   
                    if acc > best_acc:
                        torch.save(clip_net.state_dict(), args.model_dir + f'{args.clip_img_head}_epoch_{str(i)}.pth')
                        best_acc = acc
                    logger.info('acc:{} in step:{},loss:{},lr{}'.format(str(acc),str(i * len(data_loader) + idx + 1), str(float(loss)), str(optimizer.state_dict()['param_groups'][0]['lr'])))
            scheduler.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=r'./model/',
                        help='path to model')
    parser.add_argument('--img_ch_dir', type=str, default=r'/nas_data/WTY/dataset/visualC3/char/',
                        help='path to datasets')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test'],
                        help='train,valid or test')
    parser.add_argument('--batch_size', type=int, default=128, help='the size of a batch')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'])
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.98))
    parser.add_argument('--eps', type=float, default=1e-06)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--valid_step', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--clip_img_head', type=str, default='RN50',
                        choices=['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50'])
    parser.add_argument('--model_mode', type=str, default='common',
                        choices=['clip', 'common'])
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
