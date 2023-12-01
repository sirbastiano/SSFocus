import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset
import configparser

from Compressor import Focalizer
from Losses import shannon_entropy_loss

# init device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using {torch.cuda.get_device_name()} for training.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU for training.")
    
torch.set_float32_matmul_precision('high')  # For performance

class FocusPlModule(pl.LightningModule):
    def __init__(self, 
                input_img, 
                gt, 
                model, 
                learning_rate_a0 = 5e2, 
                learning_rate_ar = 1e-3, 
                learning_rate_an = 1e-5, 
                loss = shannon_entropy_loss):
        super().__init__()

        self.input_img = input_img
        self.gt = gt
        self.model = model
        # setting different lr parameters  
        self.lr = 1e-3      
        self.lr_a0 = learning_rate_a0
        self.lr_ar = learning_rate_ar
        self.lr_an = learning_rate_an
        
        self.loss_fn = loss

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Creating parameter groups by filtering the parameters
        a0_params = [p for n, p in self.named_parameters() if 'a0' in n]
        ar_params = [p for n, p in self.named_parameters() if 'ar' in n]
        an_params = [p for n, p in self.named_parameters() if 'an' in n]
        other_params = [p for n, p in self.named_parameters() if 'a0' not in n and 'ar' not in n and 'an' not in n]

        # RMSprop
        # AdamW
        # Adam 
        # Adagrad
        optimizer = torch.optim.Adagrad([
            {'params': a0_params, 'lr': self.lr_a0},
            {'params': ar_params, 'lr': self.lr_ar},
            {'params': an_params, 'lr': self.lr_an},
            {'params': other_params, 'lr': self.lr}
        ])
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        return {
                    'optimizer': optimizer,
                    'lr_scheduler': scheduler,
                    'monitor': 'train_loss',  # optional key used for early stopping
                }
        
    def train_dataloader(self):
        assert self.input_img is not None, "Input image tensor is not initialized"
        if self.gt is not None:
            assert self.input_img.shape[0] == self.gt.shape[0], "Size mismatch between input and ground truth"
            dataset = TensorDataset(self.input_img, self.gt)
        else:
            dataset = TensorDataset(self.input_img.unsqueeze(0))
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    
    def training_step(self, batch, batch_idx):
        #img, gt = batch
        img = batch
        img = img[0] # hopper ... since gt is none
        output = self.model(img)
        #if gt is not None:
        #    loss = self.loss_fn(output, gt)
        #else:
        #    loss = self.loss_fn(output)
        loss = self.loss_fn(output)
        params = list(self.model.parameters())
        
        print('\n')
        a0, ar, an = params
        print('a0', a0, '\n ar', ar, '\n an', an)
        print('train_loss', loss)
        print('\n')
        
        self.log('train_loss', loss)
        self.log('alfa0', a0)
        self.log('alfa_r', ar)
        self.log('alfa_n', an)
        self.log('lr', self.lr)
        return loss



if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("model_setting.ini")
    raw_path = '/media/warmachine/DBDISK/SSFocus/data/Processed/Sao_Paolo/raw_s1b-s6-raw-s-vv-20210103t214313-20210103t214344-024995-02f995.pkl'
    focus_path = '/media/warmachine/DBDISK/SSFocus/data/Processed/Sao_Paolo/focused_raw_s1b-s6-raw-s-vv-20210103t214313-20210103t214344-024995-02f995.pkl'

    raw = pd.read_pickle(raw_path)
    focused = pd.read_pickle(focus_path)

    echo = raw['echo']
    aux = raw['metadata']
    eph = raw['ephemeris']
    
    radar_data = echo[5000:9000, 2000:4000]    
    
    x = torch.tensor(radar_data, device=device).to(torch.complex128).unsqueeze(0).unsqueeze(0)
    y = torch.tensor(radar_data, device=device).to(torch.complex128).unsqueeze(0).unsqueeze(0)
    
    
    
    model = Focalizer(metadata={'aux':aux, 'ephemeris':eph}).to(device)
    print('Model loaded successfully.')

    focused_model = FocusPlModule(
                input_img=x, 
                gt=y, 
                model=model,
                learning_rate_a0 = 5e1, 
                learning_rate_ar = 1e-4, 
                learning_rate_an = 1e-5, 
                loss=shannon_entropy_loss)
    trainer = pl.Trainer(max_epochs=50, log_every_n_steps=1)
    trainer.fit(focused_model)
    
    print('Model trained successfully.')
    print('final results')
    print(list(model.parameters()))
    
    model.eval()  # Set the model to evaluation mode
    # Perform inference
    with torch.no_grad():
        output_tensor = model(x) 
        output_tensor = output_tensor.squeeze(0).squeeze(0)
        output_tensor = output_tensor.cpu().numpy()  # Removing batch dimension and sending to cpu
    # model._plot_tensor(input_tensor)
    model._plot_tensor(output_tensor, save=True, name='foc_pred.png')
    model._plot_tensor(y.squeeze(0).squeeze(0), save=True, name='gt.png')
    