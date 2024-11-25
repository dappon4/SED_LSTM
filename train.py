import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm  # Add tqdm import

from dataset import URBAN_SED
from model import SED_LSTM

EPOCH = 50
LR = 0.001

data_trasnform = lambda x: x.to('cuda')
label_transform = lambda x: x.to('cuda')

train_dataloader = DataLoader(URBAN_SED('../datasets/URBAN_SED/URBAN-SED_v2.0.0', split='train', transform=data_trasnform, target_transform=label_transform), batch_size=32, shuffle=True)
validate_dataloader = DataLoader(URBAN_SED('../datasets/URBAN_SED/URBAN-SED_v2.0.0', split='validate', transform=data_trasnform, target_transform=label_transform), batch_size=32, shuffle=True)
test_dataloader = DataLoader(URBAN_SED('../datasets/URBAN_SED/URBAN-SED_v2.0.0', split='test', transform=data_trasnform, target_transform=label_transform), batch_size=32, shuffle=True)

model = SED_LSTM(128, 256, 128, 11).to('cuda')
loss_fn = nn.BCEWithLogitsLoss()
optimzer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCH):
    # Training loop with tqdm
    train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCH} - Training")
    for i, (spectrogram, label) in enumerate(train_loader):
        #  print(label.shape) # (batch, 11, seq_len)
        optimzer.zero_grad()
        output = model(spectrogram)
        # print(output.shape) # (batch, 11, seq_len)
        loss = loss_fn(output, label)
        train_loader.set_postfix(loss=loss.item())
        loss.backward()
        optimzer.step()
        
    train_loss = loss.item()
    
    # Validation loop with tqdm
    validate_loader = tqdm(validate_dataloader, desc=f"Epoch {epoch+1}/{EPOCH} - Validation")
    for i, (spectrogram, label) in enumerate(validate_loader):
        with torch.no_grad():
            output = model(spectrogram)
            loss = loss_fn(output, label)
            validate_loader.set_postfix(loss=loss.item())
        
    validate_loss = loss.item()
    
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Validate Loss: {validate_loss}")

