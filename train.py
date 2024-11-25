import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm  # Add tqdm import
from torch.utils.tensorboard import SummaryWriter  # Add SummaryWriter import

from dataset import URBAN_SED
from model import SED_LSTM
from util import save_output

EPOCH = 50
LR = 0.001
BATCH_SIZE = 32
LOAD_ALL_DATA = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataloader = DataLoader(URBAN_SED('../datasets/URBAN_SED/URBAN-SED_v2.0.0', split='train', load_all_data=LOAD_ALL_DATA), batch_size=BATCH_SIZE, shuffle=True)
validate_dataloader = DataLoader(URBAN_SED('../datasets/URBAN_SED/URBAN-SED_v2.0.0', split='validate', load_all_data=LOAD_ALL_DATA), batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(URBAN_SED('../datasets/URBAN_SED/URBAN-SED_v2.0.0', split='test', load_all_data=LOAD_ALL_DATA), batch_size=BATCH_SIZE, shuffle=True)

model = SED_LSTM(128, 256, 128, 11).to('cuda')
loss_fn = nn.BCEWithLogitsLoss()
optimzer = torch.optim.Adam(model.parameters(), lr=LR)

writer = SummaryWriter()  # Initialize SummaryWriter

for epoch in range(EPOCH):
    train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCH} - Training")
    for i, (spectrogram, label) in enumerate(train_loader):
        
        spectrogram = spectrogram.to(device)
        label = label.to(device)
        
        optimzer.zero_grad()
        output = model(spectrogram)

        loss = loss_fn(output, label)
        train_loader.set_postfix(loss=loss.item())
        loss.backward()
        optimzer.step()
        
    train_loss = loss.item()
    
    # Validation loop with tqdm
    validate_loader = tqdm(validate_dataloader, desc=f"Epoch {epoch+1}/{EPOCH} - Validation")
    for i, (spectrogram, label) in enumerate(validate_loader):
        
        spectrogram = spectrogram.to(device)
        label = label.to(device)
        
        with torch.no_grad():
            output = model(spectrogram)
            loss = loss_fn(output, label)
            validate_loader.set_postfix(loss=loss.item())
    
    validate_loss = loss.item()

    # visualize one of the validation output
    fig = save_output(spectrogram, output, label, epoch)
    writer.add_scalar('Train Loss', train_loss, epoch+1)  # Log train loss
    writer.add_scalar('Validation Loss', validate_loss, epoch+1)  # Log validation loss
    writer.add_figure("Figure", fig, epoch+1)  # Log output figure

# save the model
torch.save(model.state_dict(), 'model.pth')
writer.close()  # Close the SummaryWriter