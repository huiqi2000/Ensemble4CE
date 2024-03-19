from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import torch.utils
import torch.utils.data
from torchvision import transforms
from PIL import Image
from mae import models_vit
from mae.util.pos_embed import interpolate_pos_embed
import torch
import os
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ensemble import RecorderMeter
import random
import numpy as np
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(101)


def pil_loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    img = Image.open(path)
    return img.convert("RGB")


def build_transform(input_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)





class InferDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        datadir = glob(os.path.join('dataset/C-EXPR-DB', '*'))
        datadir.sort()
        self.transform = build_transform(224)
        self.imgs = []

        for vid in datadir:
            imgs = glob(os.path.join(vid, '*.bmp'))
            imgs.sort(key=lambda x: int(x[-9:-4]))
            self.imgs.extend(imgs)
    
    def __getitem__(self, index):
        path = self.imgs[index]
        data = self.transform(pil_loader(path))
        p = os.path.split(path)
        
        name = p[0][-2:] + '/' + p[1][-9:-4] + '.jpg'
        return name, data

    def __len__(self):
        return len(self.imgs)





from ensemble import EnsembleDataset, EnsembleModel
class PredDataset(EnsembleDataset):
    def __init__(self, mode, args, meta=None) -> None:
        super().__init__(mode, args, meta)

        datadir = glob(os.path.join('dataset/C-EXPR-DB', '*'))
        datadir.sort()
        self.imgs = []

        for vid in datadir:
            imgs = glob(os.path.join(vid, '*.bmp'))
            imgs.sort(key=lambda x: int(x[-9:-4]))
            self.imgs.extend(imgs)

    
    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.readimg(path)
        mimg = self.manetransform(img)
        rimg = self.resnetransform(img)
        vimg = self.vitransform(img)
        p = os.path.split(path)
        
        name = p[0][-2:] + '/' + p[1][-9:-4] + '.jpg'
        return name, mimg, vimg, rimg
    
    def __len__(self):
        return len(self.imgs)
    
    

class PredModel(EnsembleModel):
    def __init__(self, args, frozen) -> None:
        super().__init__(args, frozen)
    

    @torch.no_grad()
    def forward_branch(self, mimg, vimg, rimg):
        logit_m = self.manet(mimg)
        logit_m = 0.6 * logit_m[0] + 0.4 * logit_m[1]
        logit_v = self.vit(vimg)
        logit_r = self.resnet(rimg).squeeze()

        return logit_m, logit_v, logit_r




def write_pred(path, pred):
    imgnetclass2idx = {'Angrily Surprised': 0, 'Disgustedly Surprised': 1, 'Fearfully Surprised': 2, 'Happily Surprised': 3, 'Sadly Angry': 4, 'Sadly Fearful': 5, 'Sadly Surprised': 6}
    imgnetidx2class = {v:k for k, v in imgnetclass2idx.items()}
    mapdict = {
        'Fearfully Surprised':0,
        'Happily Surprised':1,
        'Sadly Surprised':2,
        'Disgustedly Surprised':3,
        'Angrily Surprised':4,
        'Sadly Fearful':5,
        'Sadly Angry':6
    }

    with open(path, 'w') as f:
        f.write('image_location,Fearfully_Surprised,Happily_Surprised,Sadly_Surprised,Disgustedly_Surprised,Angrily_Surprised,Sadly_Fearful,Sadly_Angry'+'\n')
        for name, i in pred:
            f.write(name + ',' + str(mapdict[imgnetidx2class[i]]) + '\n')



def main():
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
    parser.add_argument('--batch_size', default=128)
    args = parser.parse_args()

    model = PredModel(args, True)
    test_set = PredDataset(mode='val', args=args, meta=model.meta)
    test_loader = DataLoader(test_set, batch_size=64, num_workers=8)
    
    model.load_state_dict(torch.load('best.pth', map_location='cpu'))
    model.cuda()

    with torch.no_grad():

        pred_ens, pred_m, pred_v, pred_r = [], [], [], []
        model.eval()
        for name, mimg, vimg, rimg in tqdm(test_loader):
            mimg, vimg, rimg = mimg.cuda(), vimg.cuda(), rimg.cuda()
            logit_ens = model(mimg, vimg, rimg)
            print(logit_ens.shape)
            logit_m, logit_v, logit_r = model.forward_branch(mimg, vimg, rimg)

            pred_ens.extend(list(zip(name, logit_ens.argmax(dim=-1).cpu().tolist())))
            pred_m.extend(list(zip(name, logit_m.argmax(dim=-1).cpu().tolist())))
            pred_v.extend(list(zip(name, logit_v.argmax(dim=-1).cpu().tolist())))
            pred_r.extend(list(zip(name, logit_r.argmax(dim=-1).cpu().tolist())))
   
    
    # torch.save([pred_ens, pred_m, pred_v, pred_r], 'predication/pred.pth')
    
    write_pred('predication/ens.txt', pred_ens)
    write_pred('predication/m.txt', pred_m)
    write_pred('predication/v.txt', pred_v)
    write_pred('predication/r.txt', pred_r)




if __name__ == '__main__':
    main()
