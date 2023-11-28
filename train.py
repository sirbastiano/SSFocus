import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset

from Compressor import Focalizer
from Losses import AF_loss, shannon_entropy_loss

# init device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name()} for training.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU for training.")
    
torch.set_float32_matmul_precision('medium')  # For performance

class FocusPlModule(pl.LightningModule):
    def __init__(self, input_img, gt = None, model = None, loss = AF_loss):
        super().__init__()

        self.input_img = input_img
        self.gt = gt
        self.model = model
        self.lr = 1e-2 # learning rate
        self.loss_fn = loss

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
        if gt is not None:
            loss = self.loss_fn(output, gt)
        else:
            loss = self.loss_fn(output)
        self.log('train_loss', loss)
        self.log('lr', self.lr)
        return loss




if __name__ == '__main__':
    
    x = torch.load('/home/roberto/PythonProjects/SSFocus/Data/4096_test_fft2D.pt')
    print('Loaded image tensor from .pt file')
    # TODO: prommpt the real gt here:
    # ground_truth = torch.rand(1, 2, 4096, 4096, dtype=torch.complex64).to(device)
    ground_truth = None
    # 
    aux = pd.read_pickle('/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/numpy/s1a-s1-raw-s-vv-20200509t183227-20200509t183238-032492-03c34a_pkt_8_metadata.pkl')
    eph = pd.read_pickle('/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/numpy/s1a-s1-raw-s-vv-20200509t183227-20200509t183238-032492-03c34a_ephemeris.pkl')
    model = Focalizer(metadata={'aux':aux, 'ephemeris':eph})
    print('Model initialized')
    focused_model = FocusPlModule(input_img=x, gt=None, model=model, loss=shannon_entropy_loss)
    print('Model parameters:')
    print(list(model.parameters()))
    
    # trainer = pl.Trainer(max_epochs=2, log_every_n_steps=1)
    # trainer.fit(focused_model)
    