import torch
from torch import nn
import numpy as np
import random
import argparse
from torch import optim
import utils.misc as misc
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import sys

from torch.utils.data import DataLoader
import logging
import os
import datetime
from model import CompondEXP, CompondEXPFeature

from mae.util.datasets import build_dataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def build_optimizer_scheduler(model, args):

    if args.optim == "AdamW":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05)
    elif args.optim == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999))
    elif args.optim == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.00001)
    else:
        raise NotImplementedError
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)

    return optimizer, scheduler




def compute_CompExp_loss(pred, label):
    b, seq = pred.shape
    ce_loss = nn.CrossEntropyLoss()
    cls_loss = ce_loss(pred, label)
    return cls_loss


def train_one_epoch(model, data_loader, optimizer, device):
    model.train(True)
    total_loss = torch.zeros(1, device=device)
    for samples in tqdm(data_loader):
        optimizer.zero_grad()

        aud, vid, label = samples
        aud = aud.to(device)
        vid = vid.to(device)
        out = model(aud, vid)
        label = label.to(device)
        loss = compute_CompExp_loss(out, label)
        
        loss_value = loss.item()
        total_loss += loss_value

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()
        optimizer.step()
    
    return total_loss / len(data_loader)



@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    labels, preds = [], []
    right, total = 0, 0

    with torch.no_grad():
        for samples in data_loader:
            aud, vid, label = samples
            aud = aud.to(device)
            vid = vid.to(device)
            label = label.to(device)
            out = model(aud, vid)
            # loss = compute_CompExp_loss(out, label)
            right += sum((label == out.argmax(dim=-1))).cpu()
            total += label.shape[0]

            pred_arr = out.detach().cpu().numpy()
            label_arr = label.detach().cpu().numpy()
            preds.append(pred_arr)
            labels.append(label_arr)


    acc = right / total
    logging.info(f'acc {acc}')
    return acc


def build_model(args)->nn.Module:
    return CompondEXP(args)

def build_model_feature(args)->nn.Module:
    return CompondEXPFeature(args)


def main(args):
    device = torch.device('cuda:6')
    
    args.output_dir = args.output_dir + '/' + datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    os.makedirs(args.output_dir + '/ckpt', exist_ok=True)
    
    writer = SummaryWriter(log_dir=f'{args.output_dir}')

    logging.basicConfig(level=logging.INFO, format='%(filename)-15s[%(lineno)03d] %(message)s',
                        # datefmt='%Y-%m-%d %H:%M:%S',
                        filename=f'{args.output_dir}/train.log',
                        filemode='w')

    logging.info('\n{}\n# ------------------------------------'.format(args))

    
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)
    
    trainloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # model = build_model(args)
    model = build_model_feature(args).to(device)
    gpus = [6,7]
    model = torch.nn.DataParallel(model, device_ids=gpus)

    optimizer, scheduler = build_optimizer_scheduler(model, args)

    bestacc = 0
    for epoch in tqdm(range(0, args.epochs), ncols=100):
        loss = train_one_epoch(model, trainloader, optimizer, device)
        # print(f"Epoch {epoch}: {loss}")
        writer.add_scalar(tag="loss", # 可以暂时理解为图像的名字
                    scalar_value=loss,  # 纵坐标的值
                    global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                    )
        
        logging.info(f'epoch {epoch} loss {loss}')
        acc = evaluate(model, testloader, device)
        writer.add_scalar(tag="accuracy", # 可以暂时理解为图像的名字
                    scalar_value=torch.tensor(acc),  # 纵坐标的值
                    global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                    )
        
        if acc > bestacc:
            bestacc = acc
            torch.save(model.module.state_dict(), f'{args.output_dir}/best.pth')

        scheduler.step()
     
        torch.save(model.module.state_dict(), f'{args.output_dir}/ckpt/{epoch:03d}.pth')
      




if __name__ == '__main__':
    parser =  argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--vitName', type=str, default='vit_base_patch16')
    parser.add_argument('--vid_ckpt', type=str, default='mae/output_dir_seed101_class7/checkpoint-73.pth')
    parser.add_argument('--n_segment', type=int, default=50)
    parser.add_argument('--data_path', type=str, default='dataset/RAFDBCE7')

    args = parser.parse_args()

    
    setup_seed(args.seed)
    
    main(args)

