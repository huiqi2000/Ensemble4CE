import torch
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import os
import random
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torchvision import transforms
from PIL import Image


def pil_loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def build_transform(input_size):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
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


class MAFWDataset(Dataset):
    def __init__(self, mode, mafwanno, aud_path=None, vid_path=None, input_size=224, n_segment=100) -> None:
        super().__init__()
        self.mode = mode
        self.samples = np.load(mafwanno, allow_pickle=True).item()[self.mode]
        
        # print(self.samples)
        self.aud_path = aud_path
        self.vid_path = vid_path
        self.transform = build_transform(input_size)
        self.n_segment = n_segment
        self.emo2label = {'anger_sadness': 0, 'anger_surprise': 1, 'disgust_surprise': 2, 'fear_sadness': 3, 'fear_surprise': 4, 'happiness_surprise': 5, 'sadness_surprise': 6}
        

    def getaud(self, index):
        audpath = os.path.join(self.aud_path, self.samples[index]['clip'] + '.npy')
        audfeat = np.load(audpath)
        audfeat = torch.from_numpy(audfeat)
        
        # truncate or padding
        if len(audfeat) < self.n_segment:
            # print(len(audfeat))
            audfeat = torch.cat([audfeat, torch.zeros(self.n_segment - len(audfeat), audfeat.shape[-1])], dim=0)
        else:
            if self.mode == 'train':
                select_id = random.sample(list(range(0, len(audfeat))), self.n_segment)
                select_id.sort()
                audfeat = audfeat[select_id]
            elif self.mode == 'test':
                select_id = torch.linspace(0, len(audfeat)-1, self.n_segment, dtype=int)
                audfeat = audfeat[select_id]
                
        # print(audfeat.shape)
        # assert audfeat.shape == torch.Size([100, 768])
        return audfeat
    

    def getvid(self, index):
        vidpath = os.path.join(self.vid_path, self.samples[index]['clip'])
        # print(vidpath)
        imgs = glob.glob(os.path.join(vidpath, '*.png'))
        # print(len(imgs))
        imgs.sort(key=lambda x: int(x[-8:-4]))
        # print(imgs)
        vid = [self.transform(pil_loader(img)) for img in imgs]
        
        if len(vid) < self.n_segment:
            vid.extend([torch.zeros(3, 224, 224)] * (self.n_segment - len(vid)))
            vid = torch.stack(vid)
        else:
            if self.mode == 'train':
                select_id = random.sample(list(range(0, len(vid))), self.n_segment)
                select_id.sort()
                vid = torch.stack(vid)[select_id]

            elif self.mode == 'test':
                select_id = torch.linspace(0, len(vid)-1, self.n_segment, dtype=int)
                vid = torch.stack(vid)[select_id]
            else:
                assert ValueError
        # print(vid.shape)
        # assert vid.shape == torch.Size([100, 3, 224, 224])
        return vid

    def __getitem__(self, index):
        aud = self.getaud(index)
        vid = self.getvid(index)
        emo = self.samples[index]['emo']
        # print(emo)
        label = self.emo2label[emo]
        # label = emo
        return aud, vid, label
    
    def __len__(self):
        return len(self.samples)
    





class MAFWDatasetFeature(Dataset):
    def __init__(self, mode, mafwanno, aud_path=None, vid_path=None, input_size=224, n_segment=100) -> None:
        super().__init__()
        self.mode = mode
        self.samples = np.load(mafwanno, allow_pickle=True).item()[self.mode]
        
        # print(self.samples)
        self.aud_path = aud_path
        self.vid_path = vid_path
        self.transform = build_transform(input_size)
        self.n_segment = n_segment
        self.emo2label = {'anger_sadness': 0, 'anger_surprise': 1, 'disgust_surprise': 2, 'fear_sadness': 3, 'fear_surprise': 4, 'happiness_surprise': 5, 'sadness_surprise': 6}
        

    def getaud(self, index):
        audpath = os.path.join(self.aud_path, self.samples[index]['clip'] + '.npy')
        audfeat = np.load(audpath)
        audfeat = torch.from_numpy(audfeat)
        
        # truncate or padding
        if len(audfeat) < self.n_segment:
            # print(len(audfeat))
            audfeat = torch.cat([audfeat, torch.zeros(self.n_segment - len(audfeat), audfeat.shape[-1])], dim=0)
        else:
            if self.mode == 'train':
                select_id = random.sample(list(range(0, len(audfeat))), self.n_segment)
                select_id.sort()
                audfeat = audfeat[select_id]
            elif self.mode == 'test':
                select_id = torch.linspace(0, len(audfeat)-1, self.n_segment, dtype=int)
                audfeat = audfeat[select_id]
                
        # print(audfeat.shape)
        # assert audfeat.shape == torch.Size([100, 768])
        return audfeat
    

    def getvid(self, index):
        vidpath = os.path.join(self.vid_path, self.samples[index]['clip'])

        vidpath = vidpath.replace('need', 'feature') + '.npy'
        vidfeat = np.load(vidpath)
        vidfeat = torch.from_numpy(vidfeat)
        
        if len(vidfeat) < self.n_segment:
            pad = torch.zeros((self.n_segment - len(vidfeat), 768))
            # print(vidfeat.shape, pad.shape)
            vid = torch.cat([vidfeat, pad], dim=0)

        else:
            if self.mode == 'train':
                select_id = random.sample(list(range(0, len(vidfeat))), self.n_segment)
                select_id.sort()
                vid = vidfeat[select_id]

            elif self.mode == 'test':
                select_id = torch.linspace(0, len(vidfeat)-1, self.n_segment, dtype=int)
                vid = vidfeat[select_id]
            else:
                assert ValueError
        # print(vid.shape)
        # assert vid.shape == torch.Size([100, 3, 224, 224])
        return vid

    def __getitem__(self, index):
        aud = self.getaud(index)
        vid = self.getvid(index)
        emo = self.samples[index]['emo']
        # print(emo)
        label = self.emo2label[emo]
        # label = emo
        return aud, vid, label
    
    def __len__(self):
        return len(self.samples)










if __name__  == '__main__':
    from tqdm import tqdm
    mafwDataset = MAFWDataset(mode='train', 
                              mafwanno='mafwanno.npy', 
                              aud_path='abaw/models--facebook--hubert-base-ls960-FRA',
                              vid_path='dataset/MAFW_need')
    
    train_set = MAFWDatasetFeature(mode='train', 
                              mafwanno='mafwanno.npy', 
                              aud_path='abaw/models--facebook--hubert-base-ls960-FRA',
                              vid_path='dataset/MAFW_need')
    test_set = MAFWDatasetFeature(mode='test', 
                              mafwanno='mafwanno.npy', 
                              aud_path='abaw/models--facebook--hubert-base-ls960-FRA',
                              vid_path='dataset/MAFW_need')
    print(len(train_set), len(test_set))
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=2)
    for i in (train_set):
        break

    print('--------')
    for i in (test_set):
        break