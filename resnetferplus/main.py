import os
import six
import sys
import argparse
import numpy as np

import torch
import torch._utils
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
import torchvision.transforms as transforms

import sys
sys.path.append('../../')

torch.manual_seed(101)


from torchvision import datasets

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def load_module_2or3(model_name, model_def_path):
    """Load model definition module in a manner that is compatible with
    both Python2 and Python3
    Args:
        model_name: The name of the model to be loaded
        model_def_path: The filepath of the module containing the definition
    Return:
        The loaded python module."""
    if six.PY3:
        import importlib.util
        spec = importlib.util.spec_from_file_location(model_name, model_def_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import importlib
        dirname = os.path.dirname(model_def_path)
        sys.path.insert(0, dirname)
        module_name = os.path.splitext(os.path.basename(model_def_path))[0]
        mod = importlib.import_module(module_name)
    return mod

def load_model(model_name, model_dir, pretrained_dir):
    """Load imoprted PyTorch model by name
    Args:
        model_name (str): the name of the model to be loaded
    Return:
        nn.Module: the loaded network
    """
    model_def_path = os.path.join(model_dir, model_name + '.py')
    weights_path = os.path.join(pretrained_dir, model_name + '.pth')
    mod = load_module_2or3(model_name, model_def_path)
    func = getattr(mod, model_name)
    net = func(weights_path=weights_path)
    return net

def compose_transforms(meta, train=True):
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    assert im_size[0] == im_size[1], 'expected square image size'
    
    transform_list = [transforms.Resize(256),
                      transforms.CenterCrop(size=(im_size[0], im_size[1])),
                      transforms.ToTensor()] if not train else [
                          transforms.RandomResizedCrop(224),
                          transforms.ToTensor()
                      ]
    if meta['std'] == [1, 1, 1]:  # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    return transforms.Compose(transform_list)

def get_feature(model, layer_name, image):
    bs = image.size(0)
    layer = model._modules.get(layer_name)
    if layer_name == 'fc7':
        my_embedding = torch.zeros(bs, 4096)
    elif layer_name == 'fc8':
        my_embedding = torch.zeros(bs, 7)
    elif layer_name == 'pool5' or layer_name == 'pool5_full':
        my_embedding = torch.zeros([bs, 512, 7, 7])
    elif layer_name == 'pool4':
        my_embedding = torch.zeros([bs, 512, 14, 14])
    elif layer_name == 'pool3':
        my_embedding = torch.zeros([bs, 256, 28, 28])
    elif layer_name == 'pool5_7x7_s1':  # available
        my_embedding = torch.zeros([bs, 2048, 1, 1])
    elif layer_name == 'conv5_3_3x3_relu': # available
        my_embedding = torch.zeros([bs, 512, 7, 7])
    else:
        raise Exception(f'Error: not supported layer "{layer_name}".')

    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    h = layer.register_forward_hook(copy_data)
    _ = model(image)
    h.remove()
    if layer_name == 'pool5' or layer_name == 'conv5_3_3x3_relu':
        GAP_layer = nn.AvgPool2d(kernel_size=[7, 7], stride=(1, 1))
        my_embedding = GAP_layer(my_embedding)

    my_embedding = F.relu(my_embedding.squeeze())
    if my_embedding.size(0) != bs:
        my_embedding = my_embedding.unsqueeze(0)

    
    return my_embedding

def extract(data_loader, model, layer_name):
    model.eval()
    with torch.no_grad():
        features, timestamps = [], []
        for imgs, ids in data_loader:
            imgs = imgs.cuda()
            batch_features = get_feature(model, layer_name, imgs)
            features.extend(batch_features)
            timestamps.extend(ids)
        features, timestamps = np.array(features), np.array(timestamps)
        return features, timestamps


if __name__ == "__main__":
    # --dataset=MER2023 
    # --feature_level='UTTERANCE' 
    # --model_name='senet50_ferplus_dag'  
    # --gpu=0
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--data',       type=str, default='/home/data02/zjc/abaw/dataset/Unity',        help='input dataset')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', help='feature level [FRAME or UTTERANCE]')
    parser.add_argument('--model_name',    type=str, default='resnet50_ferplus_dag',        choices=['resnet50_ferplus_dag', 'senet50_ferplus_dag'])
    parser.add_argument('--layer_name',    type=str, default='conv5_3_3x3_relu', help='which layer used to extract feature')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=128)
    

    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    # load pre-trained model
    PATH_TO_PRETRAINED_MODELS = '.'
    pretrained_dir = os.path.join(PATH_TO_PRETRAINED_MODELS, 'ferplus') # directory of pre-trained models
    model_dir = './pytorch-benchmarks/model'
    model = load_model(args.model_name, model_dir, pretrained_dir)
    meta = model.meta
    device = torch.device("cuda")
    model = torch.nn.parallel.DataParallel(model).cuda()
    # model = model.to(device)
    model.load_state_dict(torch.load('best_bak_unity.pth'))

    # transform
    train_transform = compose_transforms(meta, train=True)
    val_transform = compose_transforms(meta, train=False)

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_dataset = datasets.ImageFolder(traindir,
                                         train_transform)

    test_dataset = datasets.ImageFolder(valdir,
                                        val_transform)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

   
    celoss = torch.nn.CrossEntropyLoss()
    dropout = torch.nn.Dropout(0.2)

    bestacc, bestepoch = 0, 0
    from tqdm import tqdm
    for epoch in range(100):
        
        # model.train()
        # right, total = 0, 0
        # for img, label in tqdm(train_loader):
        #     # print(img.shape)
        #     optimizer.zero_grad()
        #     # img = img.to(device)
        #     # label = label.to(device)
        #     img = img.cuda()
        #     label = label.cuda()
        #     logits = dropout(model(img))
        #     logits = logits.squeeze()
        #     loss = celoss(logits, label)
        #     loss.backward()
        #     optimizer.step()

        #     right += sum(label == logits.argmax(dim=-1))
        #     total += img.shape[0]
        
        model.eval()
        trues, preds = [], []
        with torch.no_grad():
            right, total = 0, 0
            for img, label in val_loader:
                # img = img.to(device)
                # label = label.to(device)
                img = img.cuda()
                label = label.cuda()
                logits = model(img)
                logits = logits.squeeze()
                right += sum(label == logits.argmax(dim=-1))
                total += img.shape[0]


                trues.extend(label.cpu().detach().numpy())
                preds.extend(logits.cpu().argmax(dim=-1).detach().numpy())
    
            torch.save([trues, preds], 'rafdb_pred.pth')
            exit()
            
            acc = right / total
            print(f'acc {acc * 100:.5f}%')
            if acc >= bestacc:
                bestacc = acc
                bestepoch = epoch
                torch.save(model.state_dict(), 'best.pth')
            print(acc)
        scheduler.step(acc.item())

    print(bestacc, bestepoch)
        


