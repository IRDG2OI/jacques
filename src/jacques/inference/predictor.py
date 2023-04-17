import os
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from torch.multiprocessing import set_start_method
from importlib_resources import files
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import OrderedDict
from jacques.model_building.model_config import neural_network_settings
from jacques.model_building.layers import HeadNet, build_model
from jacques.dataloading.datamodule import DataModule
from jacques.dataloading.custom_datasets import UnlabeledDataset
from jacques.inference.output import SavePredictionsDf

try:
     set_start_method('spawn')
except RuntimeError:
    pass


class UselessImagesPredictor():
    def __init__(self, model, checkpoint_path):
                
        self.model = model
        self.model= self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = checkpoint_path
        
    def load_checkpoint(self):
        checkpoint_loaded = torch.load(self.checkpoint, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
        new_state_dict = OrderedDict()
        for k, v in checkpoint_loaded['state_dict'].items():
            name = k[6:] 
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        return self.model
        
    def predict(self, images):
        self.model = self.load_checkpoint()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(images)
            prob = torch.sigmoid(logits).data
            prob = prob.cpu().numpy()
            prob = prob[0]
            prob = np.around(prob, decimals=3)
        if prob > 0.5:
            label = "useless"
        else:
            label = "useful"
        return prob, label
    
def classify_useless_images(folder_path, ckpt_path):
    '''
    A function that classifies images stored in a folder.
    Input:
    folder_path : the path to a folder containing images (jpg or png formats accepted).
    ckpt_path : the path to the checkpoint trained
    Output:
    df : a 3 columns dataframe (dir, image_name and label) with the predictions made by a deep learning model.
    '''
    backbone = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    
    headnet = HeadNet(bodynet_features_out = backbone.fc.in_features,
                                   head_aggregation_function = neural_network_settings['head_aggregation'],
                                   head_hidden_layers_activation_function = neural_network_settings['head_hidden_layers_activation'],
                                   head_normalization_function = neural_network_settings['head_normalization'],
                                   head_proba_dropout = neural_network_settings['proba_dropout'],
                                   nb_hidden_head_layers = neural_network_settings['nb_hidden_head_layers'], 
                                   nb_classes = len(neural_network_settings['class_labels']))
    
    model = build_model(backbone, headnet)

    predictor = UselessImagesPredictor(model, ckpt_path)
    
    unlabeled_img = os.listdir(folder_path)
    unlabeled_img = [f for f in unlabeled_img if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.JPG') or f.endswith('.JPEG') or f.endswith('.png') or f.endswith('.PNG')]
    
    dm = DataModule()

    unlabeled_set = UnlabeledDataset(unlabeled_img, folder_path, dm.transforms_test)

    predict_dataloader = DataLoader(unlabeled_set, batch_size=1, shuffle = False, num_workers = os.cpu_count())

    df = pd.DataFrame(columns = ['dir', 'image', 'class'])

    #predict classes for selected samples
    for batch in tqdm(predict_dataloader):
        image_name, image = batch
        prediction, label = predictor.predict(image)
        row = [folder_path, image_name[0], label]
        df_length = len(df)
        df.loc[df_length] = row
    
    return df







    
