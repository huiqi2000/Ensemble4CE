import torch
from torch import nn
from mae import models_vit
import time


class temporal_lstm(nn.Module):
    def __init__(self, n_segment, input_size, output_size, hidden_size, num_layers, last="avg"):
        super(temporal_lstm, self).__init__()
        self.n_segment = n_segment
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers,  bidirectional=True ,dropout = 0.3)
        self.linear = nn.Linear(hidden_size*2, output_size)
        self.activ = nn.LeakyReLU(0.1) 
        self.last = last

    def forward(self, x):
        n_batch, t, c = x.shape
        new_x = x.view(n_batch, t, c).permute(1,0,2).contiguous()
        new_x, _ = self.gru(new_x)
        new_x = self.activ(new_x)
        new_x = new_x.view(n_batch*t , 2*self.hidden_size)
        new_x = self.linear(new_x)
        new_x = self.activ(new_x)
        new_x = new_x.view(n_batch, t, -1)
        # if self.last == "avg":
        #     new_x = torch.mean(new_x,dim=1)
        return new_x

class TransEncoder(nn.Module):
    def __init__(self, inc=512, outc=512, dropout=0.6, nheads=1, nlayer=4):
        super(TransEncoder, self).__init__()
        self.nhead = nheads
        self.d_model = outc
        self.dim_feedforward = outc
        self.dropout = dropout
        self.conv1 = nn.Conv1d(inc, self.d_model, kernel_size=1, stride=1, padding=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)

    def forward(self, x):
        out = self.conv1(x)
        out = out.permute(2, 0, 1)
        out = self.transformer_encoder(out)
        return out
    

feat_dims = {
    'hubert_base': 768,
    'hubert_large': 768,
    'wav2vec2': 1024
}

class vit_MAE(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = getattr(models_vit, args.vitName)(
                        global_pool=True,
                        num_classes=7,
                        drop_path_rate=0.1,
                        img_size=224
                        )
        if args.vid_ckpt:
            vid_ckpt = torch.load(args.vid_ckpt, map_location='cpu')
            self.model.load_state_dict(vid_ckpt, strict=False)
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            pass

    
    def feature(self, imgs):
        feature = self.model.forward_features(imgs)
        return feature



class CompondEXP(nn.Module):
    def __init__(self, args):
        super(CompondEXP, self).__init__()
        self.vid_extractor = vit_MAE(args)

        # if config["use_audio_fea"] == True:
        #     self.audio_temporal = temporal_lstm(config["n_segment"], input_size=768, output_size=512, hidden_size=512, num_layers=4, last="identity")
        
        # concat_dim = 768
        # if config["use_audio_fea"] == True:
        #     concat_dim += config['audio_fea_dim']
            
        concat_dim = 768
        
        hidden_size1,hidden_size2,hidden_size3 = 256,128,64
        
        self.feat_fc = nn.Conv1d(concat_dim, hidden_size1, 1, padding=0)
        self.activ = nn.LeakyReLU(0.1) 
        self.dropout = nn.Dropout(p=0.3)
        self.head = nn.Sequential(
                nn.Linear(hidden_size2, hidden_size3),
                nn.BatchNorm1d(hidden_size3),
                nn.Linear(hidden_size3, 7)
                )
       
        self.transformer = TransEncoder(inc=hidden_size1, outc=hidden_size2, dropout=0.3, nheads=4, nlayer=4)
        
    
    def forward_feature(self, aud, vid):
        bs, seq, c, h, w = vid.shape
        vid = vid.view(bs*seq, c, h, w)
        with torch.no_grad():
            vid_fea= self.vid_extractor.feature(vid)  # (bl, d)
        
        vid_fea = vid_fea.view(bs, seq, -1)
        # concats = [aud, vid_fea]
        # concat_fea = torch.cat(concats,dim=2)          # (b, l, d')
        concat_fea = vid_fea
        feat = torch.transpose(concat_fea, 1, 2) # (b, d', l)
        feat = self.feat_fc(feat) # (b, d', l)
        feat = self.activ(feat) # (b, d', l)
        out = self.transformer(feat) # (l, b, d')
        out = torch.transpose(out, 1, 0) #(b, l, d')
        out = torch.reshape(out, (bs*seq, -1)) #(bl, d')
        out = self.head(out) # (bl, 7)
        out = out.view(bs,seq,-1) # (b, l, 7)
        return out

    def forward(self, aud, vid):
        # out = self.forward_feature(aud, vid).mean(dim=1)
        out = self.forward_feature(aud, vid)[:, 0, :]
        return out






class CompondEXPFeature(nn.Module):
    def __init__(self, args):
        super(CompondEXPFeature, self).__init__()
        concat_dim = 768
        
        hidden_size1,hidden_size2,hidden_size3 = 256,128,64
        
        self.feat_fc = nn.Conv1d(concat_dim, hidden_size1, 1, padding=0)
        self.activ = nn.LeakyReLU(0.1) 
        self.dropout = nn.Dropout(p=0.3)
        self.head = nn.Sequential(
                nn.Linear(hidden_size2, hidden_size3),
                nn.BatchNorm1d(hidden_size3),
                nn.Linear(hidden_size3, 7)
                )
       
        self.transformer = TransEncoder(inc=hidden_size1, outc=hidden_size2, dropout=0.3, nheads=4, nlayer=4)
        
    
    def forward_feature(self, aud, vid):
        bs, seq, _ = vid.shape
        vid_fea = vid.view(bs, seq, -1)
        # concats = [aud, vid_fea]
        # concat_fea = torch.cat(concats,dim=2)          # (b, l, d')
        concat_fea = vid_fea
        feat = torch.transpose(concat_fea, 1, 2) # (b, d', l)
        feat = self.feat_fc(feat) # (b, d', l)
        feat = self.activ(feat) # (b, d', l)
        out = self.transformer(feat) # (l, b, d')
        out = torch.transpose(out, 1, 0) #(b, l, d')
        out = torch.reshape(out, (bs*seq, -1)) #(bl, d')
        out = self.head(out) # (bl, 7)
        out = out.view(bs,seq,-1) # (b, l, 7)
        return out

    def forward(self, aud, vid):
        # out = self.forward_feature(aud, vid).mean(dim=1)
        out = self.forward_feature(aud, vid).mean(dim=1)
        return out










if __name__ == "__main__":
    from dataset_loader import MAFWDataset

    train_set = MAFWDataset(mode='train', 
                            mafwanno='mafwanno.npy', 
                            aud_path='abaw/models--facebook--hubert-base-ls960-FRA',
                            vid_path='dataset/MAFW_need')
    
    test_set = MAFWDataset(mode='test', 
                            mafwanno='mafwanno.npy', 
                            aud_path='abaw/models--facebook--hubert-base-ls960-FRA',
                            vid_path='dataset/MAFW_need')



    vid = torch.randn(16, 100, 3, 224, 224).cuda(1)
    aud = torch.randn(16, 100, 768).cuda(1)

    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument('--vitName', type=str, default='vit_base_patch16')
    parser.add_argument('--vid_ckpt', type=str, default='mae/output_dir_base/checkpoint-0.pth')
    
    args = parser.parse_args()

    model = CompondEXP(args).cuda(1)

    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)

    print('loaded...')
    with torch.no_grad():
        y = model(aud, vid)
    print(y.shape)
    import time
    time.sleep(60)