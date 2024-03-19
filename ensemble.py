import torch
import torch.utils
import torch.utils.data
from manet.model.manet import manet
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from mae.util.datasets import build_transform
from mae import models_vit
from mae.util.pos_embed import interpolate_pos_embed
from resnetferplus.main import compose_transforms, load_model

import time
import os
import glob
from PIL import Image
from resnetferplus.main import get_feature
from torch import nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("log")      #第一个参数指明 writer 把summary内容 写在哪个目录下

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(101)


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")



class EnsembleDataset(torch.utils.data.Dataset):
    def __init__(self, mode, args, meta=None) -> None:
        super().__init__()
        self.samples = []
        emos = os.listdir(os.path.join(args.data, f'{mode}'))
        emos.sort()
        for idx, emo in enumerate(emos):
            self.samples.extend([_, idx] for _ in glob.glob(os.path.join(args.data, f'{mode}', emo, '*')))
        
        self.readimg = pil_loader

        if mode == 'train':
            self.manetransform = transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
            self.vitransform = build_transform(True, args)
            self.resnetransform = compose_transforms(meta, train=True)
    
        else:

            self.manetransform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            self.vitransform = build_transform(False, args)
            self.resnetransform = compose_transforms(meta, train=False)



    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.readimg(path)
        
        mimg = self.manetransform(img)
        vimg = self.vitransform(img)
        rimg = self.resnetransform(img)

        return mimg, vimg, rimg, torch.tensor(label)
    
    
    def __len__(self):
        return len(self.samples)






class EnsembleModel(torch.nn.Module):
    def __init__(self, args, frozen) -> None:
        super().__init__()

        # -------------------------------------------------------------------------
        self.manet = manet(num_classes=7)
        checkpoint = torch.load('/home/data02/zjc/abaw/manet/checkpoint/[03-18]-[04-04]-model_best.pth', map_location='cpu') # on ce7
        pre_trained_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.manet.load_state_dict(pre_trained_dict)
        if frozen:
            for p in self.manet.parameters():
                p.requires_grad_(False)
        

        # -------------------------------------------------------------------------
        self.vit = getattr(models_vit, 'vit_base_patch16')(
                    global_pool=True,
                    num_classes=7,
                    drop_path_rate=0.1,
                    img_size=224
                    )
        checkpoint = torch.load('/home/data02/zjc/abaw/mae/output_dir_rafdbce7/checkpoint-46.pth', map_location='cpu')['model']
        interpolate_pos_embed(self.vit, checkpoint)
        self.vit.load_state_dict(checkpoint)
        if frozen:
            for p in self.vit.parameters():
                p.requires_grad_(False)


        # -------------------------------------------------------------------------
        PATH_TO_PRETRAINED_MODELS = 'resnetferplus'
        pretrained_dir = os.path.join(PATH_TO_PRETRAINED_MODELS, 'ferplus') # directory of pre-trained models
        model_dir = 'resnetferplus/pytorch-benchmarks/model'
        self.resnet = load_model('resnet50_ferplus_dag', model_dir, pretrained_dir)
        
        self.meta = self.resnet.meta
        pre_trained_dict = {k.replace('module.', ''): v for k,v in torch.load('resnetferplus/best.pth', map_location='cpu').items()}
        self.resnet.load_state_dict(pre_trained_dict, strict=True)


        self.head = nn.Sequential(nn.Linear(2304, 2304//2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(2304//2, 7), nn.Dropout(0.1))

    
    def forward(self, mimg, vimg, rimg):
        with torch.no_grad():
            f1 = self.manet(mimg, return_embedding=True)
            # print(f1.device)
            f2 = self.vit.forward_features(vimg)
            # print(f2.device)
            f3 = get_feature(self.resnet, 'conv5_3_3x3_relu', rimg).cuda()
            # print(f3.device)
        
        f = torch.cat([f1, f2, f3], dim=-1)
   
        return self.head(f)

    def forward_feature(self, mimg, vimg, rimg):
        with torch.no_grad():
            f1 = self.manet(mimg, return_embedding=True)
            # print(f1.device)
            f2 = self.vit.forward_features(vimg)
            # print(f2.device)
            f3 = get_feature(self.resnet, 'conv5_3_3x3_relu', rimg).cuda()
            # print(f3.device)
        
        f = torch.cat([f1, f2, f3], dim=-1)
        return f


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/data02/zjc/abaw/dataset/RAFDBCE7')
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()
    

    
    model = EnsembleModel(args, True)
    model.load_state_dict(torch.load('best.pth', map_location='cpu'))
    model.cuda()



    train_dataset = EnsembleDataset('train', args, model.meta)
    test_dataset = EnsembleDataset('val', args, model.meta)


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True)



    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True, min_lr=5e-7)

   
    celoss = torch.nn.CrossEntropyLoss()
    dropout = torch.nn.Dropout(0.2)

    bestacc, bestepoch = 0, 0
    from tqdm import tqdm


    # eval

    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        right, total = 0, 0
        for mimg, vimg, rimg, label in tqdm(val_loader):
            # img = img.to(device)
            # label = label.to(device)
            mimg, vimg, rimg = mimg.cuda(), vimg.cuda(), rimg.cuda()
            # img = img.cuda()
            label = label.cuda()
            logits = model(mimg, vimg, rimg)
            logits = logits.squeeze()
            right += sum(label == logits.argmax(dim=-1))
            total += label.shape[0]


            trues.extend(label.cpu().detach().numpy())
            preds.extend(logits.cpu().argmax(dim=-1).detach().numpy())

        torch.save([trues, preds], 'rafdb_ense_pred.pth')
        acc = right / total
        print(f'acc {acc * 100:.5f}%')
        exit()
        
        










    for epoch in range(100):
        
        model.train()
        right, total = 0, 0
        total_loss = 0
        for mimg, vimg, rimg, label in tqdm(train_loader):
            # print(img.shape)
            optimizer.zero_grad()
            # img = img.to(device)
            # label = label.to(device)
            # img = img.cuda()
            mimg, vimg, rimg = mimg.cuda(), vimg.cuda(), rimg.cuda()
            label = label.cuda()
            logits = model(mimg, vimg, rimg)
            logits = logits.squeeze()
            loss = celoss(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            right += sum(label == logits.argmax(dim=-1))
            total += label.shape[0]
        
        total_loss = total_loss / len(train_loader)
        writer.add_scalar(tag="loss", # 可以暂时理解为图像的名字
            scalar_value=total_loss,  # 纵坐标的值
            global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
            )
        
        
        model.eval()
        trues, preds = [], []
        with torch.no_grad():
            right, total = 0, 0
            for mimg, vimg, rimg, label in tqdm(val_loader):
                # img = img.to(device)
                # label = label.to(device)
                mimg, vimg, rimg = mimg.cuda(), vimg.cuda(), rimg.cuda()
                # img = img.cuda()
                label = label.cuda()
                logits = model(mimg, vimg, rimg)
                logits = logits.squeeze()
                right += sum(label == logits.argmax(dim=-1))
                total += label.shape[0]


                trues.extend(label.cpu().detach().numpy())
                preds.extend(logits.cpu().argmax(dim=-1).detach().numpy())
    
            # torch.save([trues, preds], 'rafdb_pred.pth')
            # exit()
            
            acc = right / total
            print(f'acc {acc * 100:.5f}%')
            writer.add_scalar(tag="acc", # 可以暂时理解为图像的名字
            scalar_value=acc,  # 纵坐标的值
            global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
            )


            if acc >= bestacc:
                bestacc = acc
                bestepoch = epoch
                torch.save(model.state_dict(), 'best.pth')
            print(acc)
        scheduler.step(acc.item())

    print(bestacc, bestepoch)