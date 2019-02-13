##########################################
# Simple autoencoder in pytorch.
# Author: simranjit2112@gmail.com

# This file implements two autoencoders, one based on geometric features of the face and
# another based on appearance features.
##########################################

import torch
import torch.nn as nn
import torch.optim as optim


class appearance_autoencoder(nn.Module):
    def __init__(self):
        super(appearance_autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        self.fc1 = nn.Sequential(                    
            nn.Linear(128*8*8, 50),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(50, 128, kernel_size=8, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # TODO: Fill in forward pass
        enc = self.encoder(x)
        enc_reshaped = enc.view(-1, 128*8*8)
        fc = self.fc1(enc_reshaped)
        fc_reshaped = fc.view(-1, 50, 1, 1)
        dec = self.decoder(fc_reshaped)
        return dec


class landmark_autoencoder(nn.Module):
    def __init__(self):
        super(landmark_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
        )
        self.decoder = nn.Sequential(
        )

    def forward(self, x):
            # TODO: Fill in forward pass
        return x_recon


class autoencoder(object):
    def __init__(self, appear_lr, landmark_lr, use_cuda):
        self.appear_model = appearance_autoencoder()
        self.landmark_model = landmark_autoencoder()
        self.use_cuda = use_cuda
        if use_cuda:
            self.appear_model.cuda()
            self.landmark_model.cuda()
        self.criterion = nn.MSELoss()
        self.appear_optim = optim.Adam(self.appear_model.parameters(), lr=appear_lr)
        self.landmark_optim = optim.Adam(self.landmark_model.parameters(), lr=landmark_lr)
        
    def train_appear_model(self, epochs, trainloader):
        print("Training appearance model.....")
        self.appear_model.train()
        epoch = 0
        for epoch in range(0, epochs):
            t_loss = 0
            print("epoch {0}".format(epoch))
            for batch, x_train in enumerate(trainloader):
                if self.use_cuda:
                    x_train = x_train.cuda()
                self.appear_optim.zero_grad()
                x_train_reconstructed = self.appear_model(x_train)
                loss = self.criterion(x_train_reconstructed, x_train)
                loss.backward()
                self.appear_optim.step()

                t_loss = t_loss + loss.item()
            print("Appearance training, epoch: {0} loss: {1}".format(epoch, t_loss/len(trainloader)))


    def train_landmark_model(self, epochs, trainloader):
        self.landmark_model.train()
        epoch = 0
        # TODO: Train landmark autoencoder

    def test_appear_model(self, testloader):
        self.appear_model.eval()
        # TODO: Test appearance autoencoder
    
    def test_landmark_model(self, testloader):
        self.landmark_model.eval()
        # TODO: Test landmark autoencoder

