from ast import Break
import os
import time
import cv2
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
import torch
from PIL import Image


def create_dataset(root="../Data"):
    imgs=[]
    labels=[]
    
    transform=transforms.Compose([
        transforms.Resize((48,48)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])
    
    classes=os.listdir(root)
    
    for i,cls in enumerate(classes):
        cls_path=os.path.join(root,cls)
        images=os.listdir(cls_path)
        
        for img_name in images[:10000]: # use 8000 for save from classes imbalance
            img_path=os.path.join(cls_path,img_name)
            
            img=cv2.imread(img_path)
            
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            pil_img = Image.fromarray(img)

            Tr_img = transform(pil_img)
            imgs.append(Tr_img)
            labels.append(i)

    x=torch.stack(imgs)
    y=torch.tensor(labels)
    torch.save(classes,"classes.pt")
    return x,y

def save_processed_dataset(root="../data", save_path="processed_dataset.pt"):
    
    x,y = create_dataset(root)

    torch.save({"X": x, "y": y}, save_path)

    print(f"\nSaved processed dataset to: {save_path}")
    
    print(f"Saved shapes: X={x.shape}, y={y.shape}")


def load_processed_dataset(dataset_path="processed_dataset.pt"):
    
    data = torch.load(dataset_path, map_location="cpu")
    
    X, y = data["X"], data["y"]
    print(f"\nLoaded dataset shapes: X={X.shape}, y={y.shape}")
    return X, y


def create_loader(dataset_path, batch_size=32):
    X, y = load_processed_dataset(dataset_path)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader


def split_data(loader):
    
    dataset = loader.dataset
    
    val_size = int(len(dataset) * 0.1)
    
    train_size = len(dataset) - val_size
    
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=loader.batch_size, shuffle=True)
    
    val_loader   = torch.utils.data.DataLoader(val_set, batch_size=loader.batch_size, shuffle=False)
    
    return train_loader , val_loader