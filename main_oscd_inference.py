from pathlib import Path
# from copy import deepcopy
from argparse import ArgumentParser
from torchvision.transforms import Compose, ToTensor
from PIL import Image

import torch
from torch.nn import BCEWithLogitsLoss
from torchvision.models.resnet import resnet18
from torchvision.models import resnet50
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Precision, Recall, F1Score

from oscd_datamodule_inference import ChangeDetectionDataModule
from segmentation import get_segmentation_model
# from models.moco2_module import MocoV2
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
import rasterio
import os
import numpy as np
from glob import glob



# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")
#device = torch.device("cpu")

# If CUDA is not available and you expect it should be, raise an error or warning
#if device.type == 'cuda':
    #print(f"Number of GPUs available: {torch.cuda.device_count()}")
#else:
    #print("CUDA is not available. Using CPU instead.")

def get_args():
    parser = ArgumentParser()
    
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--resnet_type', type=int, default=18)
    parser.add_argument('--init_type', type=str, default='random')
    parser.add_argument('--ckp_path', type=str, default=None)
    parser.add_argument('--n_channels', dest='nc', type=int, default=13)
    parser.add_argument('--n_epochs', dest='ne', type=int, default=100)
    parser.add_argument('--learning_rate', dest='lr', type=float, default=0.001)
    parser.add_argument('--value_discard', type=bool, default=True)
    parser.add_argument('--patch_size', type=int, default=511)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--m_threshold', dest='mth', type=float, default=0.5)
    parser.add_argument('--result_dir', type=str)
    
    args = parser.parse_args()
    
    return args


def dice_loss(out,mask,epsilon=1e-5):
    inter = torch.dot(out.reshape(-1), mask.reshape(-1))
    sets_sum = torch.sum(out) + torch.sum(mask)
    return (2 * inter + epsilon) / (sets_sum + epsilon)


class SiamSegment(LightningModule):

    def __init__(self, backbone, feature_indices, feature_channels):
        super().__init__()
        self.model = get_segmentation_model(backbone, feature_indices, feature_channels)
        self.criterion = BCEWithLogitsLoss()
        self.dice_loss = dice_loss
        self.prec = Precision(task='binary',threshold=args.mth)
        self.rec = Recall(task='binary',threshold=args.mth) 
        self.f1 = F1Score(task='binary',threshold=args.mth) 

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        img_1, img_2, mask, pred, loss, prec, rec, f1 = self.shared_step(batch)
        self.log('train/loss', loss, prog_bar=True,sync_dist=True)
        self.log('train/precision', prec, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log('train/recall', rec, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log('train/f1', f1, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        tensorboard = self.logger.experiment
        global_step = self.trainer.global_step
        #if args.nc == 3:
        #    tensorboard.add_image('train/img_1', img_1[0], global_step)
        #    tensorboard.add_image('train/img_2', img_2[0], global_step)
        #    tensorboard.add_image('train/mask', mask[0], global_step)
        #    tensorboard.add_image('train/out', pred[0], global_step)
        #else:
        #    tensorboard.add_image('train/img_1', img_1[0,1:4,:,:], global_step)
        #    tensorboard.add_image('train/img_2', img_2[0,1:4,:,:], global_step)
        #    tensorboard.add_image('train/mask', mask[0], global_step)
        #    tensorboard.add_image('train/out', pred[0], global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        #print("batch",batch[0].shape)
        img_1, img_2, pred = self.shared_step(batch)
        #self.log('val/loss', loss, prog_bar=True,sync_dist=True)
        #self.log('val/precision', prec, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        #self.log('val/recall', rec, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        #self.log('val/f1', f1, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        #tensorboard = self.logger.experiment
        global_step = self.trainer.global_step
        
        """
        if args.nc == 3:
            tensorboard.add_image('val/img_1', img_1[0], global_step)
            tensorboard.add_image('val/img_2', img_2[0], global_step)
            tensorboard.add_image('val/mask', mask[0], global_step)
            tensorboard.add_image('val/out', pred[0], global_step)
        else:
            tensorboard.add_image('val/img_1', img_1[0,1:4,:,:], global_step)
            tensorboard.add_image('val/img_2', img_2[0,1:4,:,:], global_step)
            tensorboard.add_image('val/mask', mask[0], global_step)
            tensorboard.add_image('val/out', pred[0], global_step)
        """

        num_images = len(img_1)  # Number of images in the batch
        #print(num_images)

        for i in range(num_images):
            if args.nc == 3:
                img1_visual = img_1[i].permute(1, 2, 0).cpu().numpy()
                img2_visual = img_2[i].permute(1, 2, 0).cpu().numpy()
            else:
                # Extract the specific bands for RGB visualization
                img1_visual = img_1[i, [3, 2, 1]].permute(1, 2, 0).cpu().numpy() * 10
                img2_visual = img_2[i, [3, 2, 1]].permute(1, 2, 0).cpu().numpy() * 10
            
            #mask_visual = mask[i].squeeze().cpu().numpy()
            pred_visual = pred[i].squeeze().cpu().numpy()

            plt.figure(figsize=(20, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(img1_visual)
            plt.title(f'Image 1 (batch {batch_idx}, img {i})')

            plt.subplot(1, 3, 2)
            plt.imshow(img2_visual)
            plt.title(f'Image 2 (batch {batch_idx}, img {i})')

            #plt.subplot(1, 4, 3)
            #plt.imshow(mask_visual, cmap='gray')
            #plt.title('Ground Truth')

            plt.subplot(1, 3, 3)
            plt.imshow(pred_visual, cmap='gray')
            plt.title('Prediction')

            plt.tight_layout()
            plt.show()
        
        
        
        exit()

        #return loss

    def shared_step(self, batch):
        img_1, img_2, meta = batch

        out = self(img_1, img_2)

        pred = torch.sigmoid(out)
        #loss = self.criterion(out, mask)
        # + self.dice_loss(out,mask)
        #prec = self.prec(pred, mask.long())
        #rec = self.rec(pred, mask.long())
        #f1 = self.f1(pred, mask.long())
        return img_1, img_2, pred # mask, pred, loss, prec, rec, f1

    def configure_optimizers(self):
        # params = self.model.parameters()
        params = set(self.model.parameters()).difference(self.model.encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [scheduler]


def load_ssl_resnet_encoder(net, ckp_path, pf_sdict='module.encoder_q.'):
    # load ckp file
    state_dict = torch.load(ckp_path, map_location='cpu')['state_dict']

    # modify keys in state_dict
    for k in list(state_dict.keys()):
        if k.startswith(pf_sdict) and not k.startswith(pf_sdict+'fc'):
            newk = k[len(pf_sdict):]
            state_dict[newk] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    #load weights
    msg = net.load_state_dict(state_dict,strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    
    # print ckp info after succussfully loading it
    #print(f"Checkpoint has been loaded from \'{ckp_path}\'!")
    
    return net

def read_image(path, normalize=True, value_discard=True):
    #print("-----------",path)
    # Read all bands from the file at once
    with rasterio.open(path) as src:
        img = src.read()
        meta = src.meta

    # Normalize the image based on provided quantile data
    if normalize:
        img = normalize_image(img, value_discard)

    # Transpose the image to shape (height, width, channels)
    return np.transpose(img, (1, 2, 0)), meta
    

def normalize_image(img, value_discard=True):
    min_q = np.array([QUANTILES_ALL['min_q'][b] for b in ALL_BANDS])
    max_q = np.array([QUANTILES_ALL['max_q'][b] for b in ALL_BANDS])

    if value_discard:
        img = (img - min_q[:, None, None]) / (max_q[:, None, None] - min_q[:, None, None])
        #print("img")
    else:
        img = img / 10000

    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img



    
ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 
             'B05', 'B06', 'B07', 'B08', 
             'B8A', 'B09', 'B10', 'B11', 
             'B12']


QUANTILES_ALL = {
    'min_q': {
        'B01': 1194.0, 'B02': 885.0,  'B03': 667.0,  'B04': 426.0,
        'B05': 392.0,  'B06': 358.0,  'B07': 349.0,  'B08': 290.0,
        'B8A': 310.0,  'B09': 96.0,   'B10': 7.0,    'B11': 155.0,
        'B12': 109.0
    },
    'max_q': {
        'B01': 2456.0, 'B02': 2620.0, 'B03': 2969.0, 'B04': 3698.0,
        'B05': 3803.0, 'B06': 3994.0, 'B07': 4261.0, 'B08': 4141.0,
        'B8A': 4435.0, 'B09': 1589.0, 'B10': 51.0,   'B11': 5043.0,
        'B12': 4238.0
    }
}
                 
# Define the transformations
transform = Compose([
    ToTensor(),  # Converts images to PyTorch tensors
    # Add other transformations here if used during training
    # e.g., Normalize(mean=[mean_values], std=[std_values])
])

def apply_transforms(*images):
    # Define the transformations
    transform = Compose([
        ToTensor(),  # Converts images to PyTorch tensors
        # Normalize or other transforms can be added here if needed
        # For example:
        # Normalize(mean=[mean_r, mean_g, mean_b], std=[std_r, std_g, std_b])
    ])

    # Apply the same transformations to all images
    transformed_images = tuple(transform(image) for image in images)
    return transformed_images
    
if __name__ == '__main__':
    pl.seed_everything(42)
    
    # args
    args = get_args()
    
    #print(args)


    # dataloader
    assert(args.nc==3 or args.nc==13 or args.nc==12)
    datamodule = ChangeDetectionDataModule(args.data_dir, RGB_bands=True if args.nc==3 else False, \
                                           BGR_bands=False, \
                                           S2A_bands=True if args.nc==12 else False, \
                                           value_discard=args.value_discard, \
                                           patch_size=args.patch_size, batch_size=args.batch_size)


    # construct backbone model
    pretrained = False
    # check whether use imagenet pretrained weights, which are only applied for RGB model
    if args.init_type == 'imagenet':
        assert(args.nc==3)
        pretrained = True
    # backbone definition
    if args.resnet_type == 18:
        backbone = resnet18(pretrained=pretrained)
        feature_channels = (64, 64, 128, 256, 512)
    elif args.resnet_type == 50:
        backbone = resnet50(pretrained=pretrained)
        feature_channels=(64, 256, 512, 1024, 2048)
    else:
        raise ValueError()
    #print(f'Construct the backbone of resnet{args.resnet_type}-initialization: {args.init_type}.')
    
    # change the number of input channels of backbone
    if args.nc != 3:
        backbone.conv1 = torch.nn.Conv2d(args.nc, 64, 7, stride=2, padding=3, bias=False)
        #print(f'Modify the number of inputs of the backbone to {args.nc}.')
    
    # fix backbone layers
    for name, param in backbone.named_parameters():
            param.requires_grad = False 

    # load ckp if given
    if args.ckp_path:
        #print("fixing backbone")
        backbone = load_ssl_resnet_encoder(backbone, args.ckp_path)
        # args.init_type = 'ssl'

    model = SiamSegment(backbone, feature_indices=(2, 4, 5, 6, 7), feature_channels=feature_channels)


    # Load the checkpoint into the model
    pathRan = "/content/drive/MyDrive/ChangeDetection/Output/ckps/random"
    checkpoint_path = glob(os.path.join(pathRan, '*.ckpt'))
    checkpoint = torch.load(checkpoint_path[0], map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    # model.example_input_array = (torch.zeros((1, 3, 96, 96)), torch.zeros((1, 3, 96, 96)))
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    #print(f"Total trainable parameters: {count_parameters(model)}")
    
    path = "/content/drive/MyDrive/ChangeDetection/Images"
    file_names = [os.path.basename(x) for x in glob(os.path.join(path+"/after", '*.tif'))]
    
    counter = 0
    for name in file_names:
    #if 1 > 0:
        #print(counter)
        counter+=1
        myPath = os.path.join(path , 'before',  name)
        img_1, meta = read_image(myPath , ALL_BANDS, value_discard=True)    # Image -> np.array, type: unit8
        
        myPath = os.path.join(path , 'after',  name)
        img_2, meta = read_image(myPath , ALL_BANDS, value_discard=True)    # Image -> np.array, type: unit8

        # Apply transformations required for the model inference
        img_1,img_2 = apply_transforms(img_1,img_2)
        img1_visual =   np.stack([img_1[3], img_1[2], img_1[1]], axis=-1) *10
        img2_visual =   np.stack([img_2[3], img_2[2], img_2[1]], axis=-1) *10
       


        img_1 = img_1.unsqueeze(0).to(device)  # Ensure data type compatibility
        img_2 = img_2.unsqueeze(0).to(device)

                
        
        out = model(img_1, img_2)
        #print("out:",out)
        pred = torch.sigmoid(out) #.detach().cpu().numpy()
        pred = pred.squeeze().detach().cpu().numpy() *100
        #print("pred:",pred.shape)
        """
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(img1_visual)
     
        plt.subplot(1, 3, 2)
        plt.imshow(img2_visual)

        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap='gray')     

        plt.show()
        """
        



        output = "/content/drive/MyDrive/ChangeDetection/Output/Inference/"
        out_file = output + name
        meta.update(dtype=rasterio.int16, count=1, compress='lzw', nodata=-9999)
        with rasterio.open(out_file, 'w', **meta) as dest:
                dest.write(pred.astype(rasterio.int32), 1) 
