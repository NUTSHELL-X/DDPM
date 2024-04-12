import torch
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from options import config_parser

parser=config_parser()
args=parser.parse_args()
ds_folder=args.dataset_path

    
class SingleFolderDataset(Dataset): #返回一个Dataset的实例，用于读取单个文件夹内的图片
    def __init__(self,folder,transform):
        self.folder=os.path.expanduser(folder)
        self.transform=transform

    def __len__(self):
        return len(os.listdir(self.folder))
    
    def __getitem__(self, idx):
        image=Image.open(os.path.join(self.folder,os.listdir(self.folder)[idx]))
        image=np.array(image)
        if self.transform:
            image=self.transform(image=image)['image']
        return image,0

class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']

def create_dataloader(res,batch_size,dataset_type):
    if isinstance(res,list) or isinstance(res,tuple):
        h,w=res
    else:
        h=res
        w=res
    # data augmentation
    transforms=A.Compose(
        [
            A.Resize(height=int(h/0.9),width=int(w/0.9)),
            A.RandomCrop(h,w),
            A.HorizontalFlip(p=0.5),
            A.RGBShift(10,10,10),
            A.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0),
            A.Normalize(mean=(0,0,0),std=(1,1,1)),
            ToTensorV2(),
        ]
    )
    if dataset_type=='image_folder':
        ds=ImageFolder(root=ds_folder,transform=Transforms(transforms))
    else:
        ds=SingleFolderDataset(folder=ds_folder,transform=transforms)
    dl=DataLoader(dataset=ds,batch_size=batch_size,shuffle=True,drop_last=True,pin_memory=True,num_workers=6)
    return dl