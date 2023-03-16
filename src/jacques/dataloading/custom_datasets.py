import os
import torch
from torch.utils.data import Dataset
from torch.multiprocessing import set_start_method
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
     set_start_method('spawn')
except RuntimeError:
    pass
    
    
class UnlabeledDataset(Dataset):
    def __init__(self, imagelist, img_dir, transform):
        self.imagepaths = imagelist
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, idx):
        imagepath = self.imagepaths[idx]
        image = Image.open(os.path.join(self.img_dir, imagepath)).convert("RGB")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.transform is not None:
            image = self.transform(image)
        image = image.to(device)
        return (imagepath, image)
    
class LabeledDataset(Dataset):
      def __init__(self, df, img_dir, transforms=None):
        self.img = df['image'].values
        self.labels = df['label'].values
        self.img_dir = img_dir
        self.transforms = transforms
    
      def __getitem__(self,idx):
        sample = self.img[idx]
        sample = Image.open(os.path.join(self.img_dir, sample)).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype= torch.float32).unsqueeze(-1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        label = label.to(device)

        if self.transforms is not None:
            sample = self.transforms(sample)
            
        sample = sample.to(device)
        return (sample, label)
  
      def __len__(self):
        return len(self.img)
       