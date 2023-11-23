import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset



from Compressor import Focalizer
from Losses import AF_loss

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name()} for training.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU for training.")
    

torch.set_float32_matmul_precision('medium')  # For performance



class FocuserModule(pl.LightningModule):
    def __init__(self, input_img, gt = None):
        super().__init__()

        self.input_img = input_img
        self.gt = gt
        self.model = 
        self.lr = 1e-2 # learning rate
        self.loss_fn = AF_loss

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
                    'optimizer': optimizer,
                    'lr_scheduler': scheduler,
                    # 'monitor': 'val_loss',  # optional key used for early stopping
                }
        
    def train_dataloader(self):
        assert self.input_img is not None, "Input image tensor is not initialized"
        if self.gt is not None:
            assert self.input_img.shape[0] == self.gt.shape[0], "Size mismatch between input and ground truth"
            dataset = TensorDataset(self.input_img, self.gt)
        else:
            dataset = TensorDataset(self.input_img)
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=7)

    
    def training_step(self, batch, batch_idx):
        img, gt = batch
        output = self.model(img)
        
        loss = self.loss_fn(output, gt)
    
        self.log('train_loss', loss)
        self.log('lr', self.lr)
        return loss




if __name__ == '__main__':
    
    x = torch.load()
    