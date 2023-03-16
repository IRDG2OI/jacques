import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from torch.multiprocessing import set_start_method
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
     set_start_method('spawn')
except RuntimeError:
    pass

class UselessImagesClassifier(pl.LightningModule):

    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model        
        self.accuracy = torchmetrics.Accuracy()
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.model(x)
        return x

    def cross_entropy_loss(self, logits, labels):
        loss = nn.BCEWithLogitsLoss()
        return loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits,y)
        acc = self.accuracy(logits, y.type(torch.int64))
        self.log("train_acc",acc,on_step=False,on_epoch=True,prog_bar=True,logger=True),
        self.log("train_loss",loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits,y.type(torch.int64))
        self.log("val_acc",acc,prog_bar=True,logger=True),
        self.log("val_loss",loss,prog_bar=True,logger=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        return [optimizer], [scheduler]
    


