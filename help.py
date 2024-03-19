# import torch
# rafidx2emo = {1: 'Surprise', 2: 'Fear',3: 'Disgust',4: 'Happiness',5: 'Sadness', 6: 'Anger' , 7: 'Neutral'}
# rafemo2idx = {emo: idx for idx, emo in rafidx2emo.items()}
# print(rafidx2emo)
# print(rafemo2idx)


import glob
import os

# r11 = glob.glob(os.path.join('dataset/RAFDBCE11', '*', '*', '*'))
# print(r11[0])
# print(len(r11))

# print('--------------------')
# r7 = glob.glob(os.path.join('dataset/RAFDBCE7', '*', '*', '*'))
# print(r7[0])
# print(len(r7))

# print('--------------------')
# print(len(os.listdir('dataset/RAFDB/compound/Image/aligned')))


# import torch
# a = torch.load('mae/output_dir_seed101_class7/checkpoint-0.pth')
# for k, v in a.items():
#     print(k, v)

# def find_classes(directory: str):
#     """Finds the class folders in a dataset.

#     See :class:`DatasetFolder` for details.
#     """
#     classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
#     if not classes:
#         raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

#     class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
#     return classes, class_to_idx

# print(find_classes('dataset/RAFDBCE7/train'))

# directory = 'dataset/MAFWCE7'
# print(find_classes(directory))

# se = os.listdir('dataset/Unity/train')
# se.sort()
# print(se)


# import numpy as np
# mafwanno = 'mafwanno.npy'
# a = np.load(mafwanno, allow_pickle=True).item()['train']
# print(a)



# from mae import models_vit
# import time

# model = getattr(models_vit, 'vit_base_patch16')(
#                         global_pool=True,
#                         num_classes=7,
#                         drop_path_rate=0.1,
#                         img_size=224
#                         )

# model.cuda()
# time.sleep(100)



a = glob.glob('/home/data02/zjc/abaw/dataset/Unity/train/*/*')
print(len(a))

