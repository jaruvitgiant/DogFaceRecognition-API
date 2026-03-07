import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image

class DogDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self): # ต้องเยื้องเข้ามา 1 ระดับจากคำว่า class
        return len(self.image_paths)

    def __getitem__(self, idx): # ต้องเยื้องเข้ามา 1 ระดับจากคำว่า class
        # โค้ดที่แก้ไขจากครั้งก่อนเพื่อให้รับได้ทั้ง Path และ PIL Image
        img_input = self.image_paths[idx]
        
        if isinstance(img_input, str):
            image = Image.open(img_input).convert("RGB")
        else:
            image = img_input.convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

def apply_aug(image, transform):
    img_np = np.array(image)
    return transform(image=img_np)['image']
class AugTransform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, img):
        return apply_aug(img, self.transform)

def get_dataloaders(train_path,image_ids, batch_size=32):
    # 1. Define Training Augmentation
    # ปรับเปลี่ยนจาก AtLeastOneBBox เป็น RandomResizedCrop เพื่อความเหมาะสมกับ Classification
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.ChannelDropout(
            channel_drop_range=[1, 1],
            fill=2
        ),
        A.ColorJitter(
            brightness=[0.8, 1.2],
            contrast=[0.8, 1.2],
            saturation=[0.8, 1.2],
            hue=[-0.5, 0.5]
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

    my_transform = AugTransform(train_transform)
    train_dataset = DogDataset(train_path, image_ids, transform=my_transform)
    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )

    num_classes = len(set(image_ids))
    return train_loader, num_classes