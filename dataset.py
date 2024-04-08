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

class MultiFolderDataset(Dataset):
    def __init__(self,folder,transform):
        folder=os.path.expanduser(folder) # change "~/" to "/home/user/"
        self.transform=transform
        self.images=[]
        self.class_idxes=[]
        class_idx=0
        for class_folder in os.listdir(folder):
            cur_folder=os.path.join(folder,class_folder)
            for image_path in os.listdir(cur_folder):
                image=Image.open(os.path.join(cur_folder,image_path))
                image=np.array(image)
                self.images.append(image)
                self.class_idxes.append(class_idx)
            class_idx+=1
        
        self.images_np_ndarray=np.ndarray(shape=(len(self.images),)+self.images[0].shape)
        for i in range(0,len(self.images)):
            self.images_np_ndarray[i] = self.images[i]
        print(self.images_np_ndarray.shape)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # image=self.images_np_ndarray[idx]
        # print('image',type(image),image.shape)
        image=self.images[idx]
        class_idx=self.class_idxes[idx]
        if self.transform:
            image=self.transform(image=image)['image']
        return image,class_idx
    
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
    elif dataset_type == 'multi_folder':
        ds=MultiFolderDataset(folder=ds_folder,transform=transforms)
    else:
        ds=SingleFolderDataset(folder=ds_folder,transform=transforms)
    dl=DataLoader(dataset=ds,batch_size=batch_size,shuffle=True,drop_last=True,pin_memory=True,num_workers=6)
    return dl