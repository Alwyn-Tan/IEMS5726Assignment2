# 1155238635 TAN Shixiang
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim


# Problem 2
def problem_2(df, k=5, t=0.2):
    list = []
    # write your logic here, test is a dataframe
    test = 0
    
    train_val, test = train_test_split(df,test_size=t, train_size=1-t)
    list = np.array_split(train_val, k)
    
    return list, test

# Problem 3
def problem_3(filename, column_label):
    # write your logic here, df is a dataframe instead
    df = pd.read_csv('problem3.csv')
    dummies = pd.get_dummies(df[column_label], dtype=int)
    df=pd.concat([df,dummies],axis=1)

    return df

# Problem 4
def problem_4(path):
    # Don't touch the settings below
    desired_length, n_fft, hop_len, n_mels, top_db = 100000, 1024, 256, 32, 80
    list_of_mel_spec_db = []
    # retrieve the list of files under path
    file_names = os.listdir(path)
    
    for file in file_names:
        # write your logic here: load audio file
        full_path = os.path.join(path, file)
        waveform, sample_rate = torchaudio.load(full_path)

        # get the current length of the waveform
        current_length = waveform.size(1)  # Size (channels, length)
    
        # write your logic here:
        # truncate the waveform, or pad the waveform with zeros
        if current_length > desired_length:
            waveform = waveform[:,:desired_length]
        else:
            zeros = torch.zeros(waveform.size(0), desired_length - current_length)
            waveform = torch.cat([waveform, zeros], dim=1)
        
        # write your logic here: define mel spec with the settings
        mel_transform = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_len,
            n_mels=n_mels,
        )
        
        # write your logic here: apply the transform to the waveform
        mel_spec = mel_transform(waveform)
        
        # write your logic here: convert to decibels with the settings
        mel_spec_db = transforms.AmplitudeToDB(top_db=top_db)(mel_spec)
        
        list_of_mel_spec_db.append(mel_spec_db)
        
    # convert the list to a tensor
    # assume all spec have the same shape
    mel_spec_tensor = torch.stack(list_of_mel_spec_db)

    return mel_spec_tensor

# Problem 5
class MyDataset(Dataset):
	def __init__(self, tensors):
		self.tensors = tensors
	def __len__(self):
		return len(self.tensors)
	def __getitem__(self, idx):
		return self.tensors[idx]

# write your autoencoder structure here
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.input_dim = 25024
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 8224),
            nn.ReLU(),
            nn.Linear(8224, 2056),
            nn.ReLU(),
            nn.Linear(2056, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2056),
            nn.ReLU(),
            nn.Linear(2056, 8224),
            nn.ReLU(),
            nn.Linear(8224, self.input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def problem_5(tensor_file):
    loaded_tensor = torch.load(tensor_file, weights_only=True)
    dataset = MyDataset(loaded_tensor)
    dataloader = DataLoader(dataset, batch_size=25, shuffle=False)
    
    model = AE()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    epochs = 20
    losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        loss1 = []
        for audio in dataloader:
            audio = audio.view(-1, 2 * 32 * 391).to(device)
            reconstructed = model(audio)[0]
            loss = loss_function(reconstructed, audio)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss1.append(loss.item())
            
        losses.append(sum(loss1) / len(loss1))
            
    return losses


class BetterAE(nn.Module):
    def __init__(self):
        super(BetterAE, self).__init__()
        self.input_dim = 25024

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 8224),
            nn.BatchNorm1d(8224),
            nn.LeakyReLU(0.2),

            nn.Linear(8224, 2056),
            nn.BatchNorm1d(2056),
            nn.LeakyReLU(0.2),

            nn.Linear(2056, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256)  # Latent Space
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 2056),
            nn.BatchNorm1d(2056),
            nn.LeakyReLU(0.2),

            nn.Linear(2056, 8224),
            nn.BatchNorm1d(8224),
            nn.LeakyReLU(0.2),

            nn.Linear(8224, self.input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Problem 6
def problem_6(tensor_file):
    # write your logic here   
    loaded_tensor = torch.load(tensor_file, weights_only=True)
    dataset = MyDataset(loaded_tensor)
    dataloader = DataLoader(dataset, batch_size=25, shuffle=False)

    model = BetterAE()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    epochs = 20
    losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        loss1 = []
        for audio in dataloader:
            audio = audio.view(-1, 2 * 32 * 391).to(device)
            reconstructed = model(audio)[0]
            loss = loss_function(reconstructed, audio)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss1.append(loss.item())

        losses.append(sum(loss1) / len(loss1))

    return losses


if __name__ == "__main__":
    # Testing: Problem 2
    df = pd.read_csv("problem2.csv")
    list, test = problem_2(df, k=5, t=0.2)
    for item in list:
        print("Segment: ", item.shape)
    print("Testing: ", test.shape)


    # Testing: Problem 3
    df = problem_3("problem3.csv","color")
    print(df)
   
   
    # Testing: Problem 4
    tensor = problem_4("TRAIN")
    print("Type of tensor:", type(tensor))
    print("Shape of tensor:", tensor.shape)
    torch.save(tensor, "problem_4.pt")

    
    # Testing: Problem 5
    losses = []
    losses = problem_5("problem_4.pt")
    i = 1
    for loss in losses:
        print(f"Epoch {i}/20, Loss: {loss:.6f}")
        i += 1
    
    
    # Testing: Problem 6
    losses = []
    losses = problem_6("problem_4.pt")
    i = 1
    for loss in losses:
        print(f"Epoch {i}/20, Loss: {loss:.6f}")
        i += 1
