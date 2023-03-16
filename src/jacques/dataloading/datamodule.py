import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path
from torch.multiprocessing import set_start_method
from jacques.dataloading.custom_datasets import LabeledDataset
try:
     set_start_method('spawn')
except RuntimeError:
    pass

    
class DataModule(pl.LightningDataModule):
    
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        
        self.transforms_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             ])
        
        self.transforms_test = transforms.Compose([transforms.Resize(224),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])

    def setup(self, stage=None):
        #Register train & test CSV
        self.train_df = pd.read_csv('/home/datawork-iot-nos/Seatizen/data/useless_classification/ground_truth_annotations/useless_useful_train80.csv')
        self.val_df = pd.read_csv('/home/datawork-iot-nos/Seatizen/data/useless_classification/ground_truth_annotations/useless_useful_test20.csv')

        self.train_df['label'] = self.train_df['label'].astype(int)
        self.val_df['label'] = self.val_df['label'].astype(int)

        # prepare transforms standards
        self.trainset = LabeledDataset(self.train_df, Path("/home/datawork-iot-nos/Seatizen/data/useless_classification/images"), self.transforms_train)

        self.valset = LabeledDataset(self.val_df, Path("/home/datawork-iot-nos/Seatizen/data/useless_classification/images"), self.transforms_test)
        

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle = True, num_workers = 8)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle = False, num_workers = 8)

    

    
