import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import datasets, transforms
from PIL import Image
import tqdm

torch.manual_seed(27)

dummy_input = torch.rand(1, 2, 256, 256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RDAnet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
            nn.Flatten(),
            nn.Linear(73728, 2014),
            nn.LeakyReLU(),
            nn.Linear(2014, 2014),
            nn.LeakyReLU(),

        )

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        self.residual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.encoder_cnn(x)
        print(x.shape)
        x = x.view(-1, 1, 19, 106)
        x = self.conv1(x)
        copy = x

        for _ in range(16):
            identity = x
            x = self.residual(x)
            x = x + identity

        x = self.conv2(x)
        x = x + copy
        x = self.conv3(x)
        x = self.conv4(x)
        return x


model = RDAnet()
print(summary(model, (2, 256, 256)))
out = model(dummy_input)
model = model.to(device)

# Params
batch_size = 32
loss_fn = torch.nn.L1Loss()
lr = 1e-4

params_to_optimize = [
    {'params': model.parameters()},
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=100,
                                                       threshold=0.0001)

class EarlyStopping:

    def __init__(self, tolerance=5, min_delta=0, verbose=False, path='checkpoint.pth'):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.verbose = verbose
        self.path = path
        self.best_score = None
        self.val_loss_min = np.Inf

    def __call__(self, validation_loss, model):

        score = -validation_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(validation_loss, model)
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(validation_loss, model)
            self.counter = 0

    def save_checkpoint(self, validation_loss, model):

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {validation_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = validation_loss


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def training_model(model, device, optimizer, loss_fn, train_loader, val_loader, scheduler):

    num_epochs = 300
    train_loss_log = []
    val_loss_log = []
    early_stopping = EarlyStopping(tolerance=50, verbose=True)

    # Train
    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        train_loss = []
        model.train()
        for x_data, y_data in train_loader:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            output_data = model(x_data)
            loss = loss_fn(output_data, y_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_batch = loss.detach().cpu().numpy()
            train_loss.append(loss_batch)

        train_loss = np.mean(train_loss)
        print('\n\n\t Training - EPOCH %d/%d - avg loss: %f\n\n' % (epoch + 1, num_epochs, train_loss))
        train_loss_log.append(train_loss)


        # Validation
        val_loss = []
        model.eval()
        with torch.no_grad():
            for x_data, y_data in val_loader:
                x_data = x_data.to(device)
                y_data = y_data.to(device)
                output_data = model(x_data)
                loss = loss_fn(output_data, y_data)
                loss_batch = loss.detach().cpu().numpy()
                val_loss.append(loss_batch)


            # Evaluate global loss
            val_loss = np.mean(val_loss)
            print('\n\n\t VALIDATION - EPOCH %d/%d - loss: %f\n\n' % (epoch + 1, num_epochs, val_loss))
            val_loss_log.append(val_loss)

        # Early stopping
        early_stopping(val_loss, model)
        scheduler.step(val_loss)

        if early_stopping.early_stop:
            print('early stop at epoch:', epoch)
            break

        print(f'Best Val Loss accuracy: {early_stopping.val_loss_min:.6f}')
        print("Current learning rate: ", optimizer.param_groups[0]['lr'])
        torch.save(model.state_dict(), 'model.pth')

    return train_loss_log, val_loss_log


# train_loss, val_loss = training_model(model, device, optimizer=optim, loss_fn=loss_fn, train_loader, val_loader, scheduler= scheduler)

