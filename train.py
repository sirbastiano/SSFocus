import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset
import configparser
import sentinel1decoder

from Compressor import Focalizer
from Losses import AF_loss, custom_loss, shannon_entropy_loss, contrast_sharpness_loss
from utility import read_zarr_database 
# init device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using {torch.cuda.get_device_name()} for training.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU for training.")
    
# torch.set_float32_matmul_precision('medium')  # For performance

class FocusPlModule(pl.LightningModule):
    def __init__(self, 
                input_img, 
                gt, 
                model, 
                learning_rate_a0 = 1e3, 
                learning_rate_ar = 1e-3, 
                learning_rate_an = 1e-3, 
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
    
    # LR = float(config["TRAINER"]["LR"])
    # raws, gt = read_zarr_database()
    # idx = 3
    # radar_data = raws[idx]
    # ground_truth = gt[idx]
    dat_path = '/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/dat/s1a-s1-raw-s-vv-20200509t183227-20200509t183238-032492-03c34a.dat'
    l0file = sentinel1decoder.Level0File(dat_path)
    
    
    # y = torch.tensor(ground_truth, device=device).to(torch.complex64).unsqueeze(0).unsqueeze(0)
    y = None
    # x = torch.tensor(radar_data, device=device).to(torch.complex64).unsqueeze(0).unsqueeze(0)
    # aux = pd.read_pickle('/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/numpy/s1a-s1-raw-s-vv-20200509t183227-20200509t183238-032492-03c34a_pkt_8_metadata.pkl')
    # eph = pd.read_pickle('/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/numpy/s1a-s1-raw-s-vv-20200509t183227-20200509t183238-032492-03c34a_ephemeris.pkl')
    selected_burst = 8
    selection = l0file.get_burst_metadata(selected_burst)
    aux = selection    
    eph = l0file.ephemeris
    # Decode the IQ data
    selection = selection[2000:3000]
    radar_data = l0file.get_burst_data(selected_burst)[2000:3000, 5000:10000]
    x = torch.tensor(radar_data, device=device).to(torch.complex64).unsqueeze(0).unsqueeze(0)
    
    model = Focalizer(metadata={'aux':aux, 'ephemeris':eph}).to(device)
    print('Model loaded successfully.')


    focused_model = FocusPlModule(
                input_img=x, 
                gt=y, 
                model=model,
                learning_rate_a0 = 5e1, 
                learning_rate_ar = 1e-4, 
                learning_rate_an = 1e-5, 
                loss=custom_loss)
    trainer = pl.Trainer(max_epochs=200, log_every_n_steps=1)
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
    model._plot_tensor(output_tensor, save=True, name='foc.png')
    # model._plot_tensor(y.squeeze(0).squeeze(0), save=True, name='gt.png')
    