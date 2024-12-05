import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # Add tqdm import
from torch.utils.tensorboard import SummaryWriter  # Add SummaryWriter import

from dataset import URBAN_SED
from model import SED_LSTM, SED_Attention_LSTM, FocalLoss
from util import save_output, save_model, clear_tmp
import time


t1 = time.time()

EPOCH = 60
LR = 0.001
BATCH_SIZE = 32
LOAD_ALL_DATA = True # change it to False if you don't have enough RAM
CHECKPOINT_STEP = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataloader = DataLoader(URBAN_SED('../datasets/URBAN_SED/URBAN-SED_v2.0.0', split='train', preprocessed_dir='n_mels_64',load_all_data=LOAD_ALL_DATA, n_mels=64), batch_size=BATCH_SIZE, shuffle=True)
validate_dataloader = DataLoader(URBAN_SED('../datasets/URBAN_SED/URBAN-SED_v2.0.0', split='validate', preprocessed_dir='n_mels_64', load_all_data=LOAD_ALL_DATA, n_mels=64), batch_size=BATCH_SIZE, shuffle=True)
# test_dataloader = DataLoader(URBAN_SED('../datasets/URBAN_SED/URBAN-SED_v2.0.0', split='test', preprocessed_dir='base', load_all_data=LOAD_ALL_DATA), batch_size=BATCH_SIZE, shuffle=True)

# model = SED_Attention_LSTM(mel_bins=64, lstm_input_size=128, hidden_size=256, num_classes=11, d_model=128).to('cuda')
model = SED_LSTM(mel_bins=64, lstm_input_size=256, hidden_size=512, num_classes=11, num_layers=3, bidirectional=True).to(device)
# loss_fn = nn.BCEWithLogitsLoss()
loss_fn = FocalLoss()
# loss_fn = nn.MSELoss()
optimzer = torch.optim.Adam(model.parameters(), lr=LR)

start_time = time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter()  # Initialize SummaryWriter
clear_tmp()  # Clear the tmp folder

for epoch in range(EPOCH):
    
    best_validate_loss = 1000
    
    train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCH} - Training", ncols=100)
    train_loss_sum = 0
    for i, (spectrogram, label) in enumerate(train_loader):
        
        spectrogram = spectrogram.to(device)
        label = label.to(device)
        
        optimzer.zero_grad()
        output = model(spectrogram)

        loss = loss_fn(output, label)
        train_loader.set_postfix(loss=loss.item())
        loss.backward()
        optimzer.step()
        train_loss_sum += loss.item()
        
    train_loss = train_loss_sum / len(train_dataloader)
    train_loader.set_postfix(loss=train_loss)
    
    # Validation loop with tqdm
    validate_loader = tqdm(validate_dataloader, desc=f"Epoch {epoch+1}/{EPOCH} - Validation", ncols=100)
    validation_loss_sum = 0
    for i, (spectrogram, label) in enumerate(validate_loader):
        
        spectrogram = spectrogram.to(device)
        label = label.to(device)
        
        with torch.no_grad():
            output = model(spectrogram)
            loss = loss_fn(output, label)
            validate_loader.set_postfix(loss=loss.item())
            validation_loss_sum += loss.item()
    
    validate_loss = validation_loss_sum / len(validate_dataloader)
    validate_loader.set_postfix(loss=validate_loss)

    # visualize one of the validation output
    fig = save_output(spectrogram, output, label, epoch)
    writer.add_scalar('Train Loss', train_loss, epoch+1)  # Log train loss
    writer.add_scalar('Validation Loss', validate_loss, epoch+1)  # Log validation loss
    writer.add_figure("Figure", fig, epoch+1)  # Log output figure
    
    if (epoch+1) % CHECKPOINT_STEP == 0:
        save_model(model, start_time, f"model-ckpt-{epoch+1}")
    
    if validate_loss < best_validate_loss:
        best_validate_loss = validate_loss
        save_model(model, start_time, f"model-best")

# save the model
save_model(model, start_time, f"model")
writer.close()  # Close the SummaryWriter
print(f"Training time: {time.time() - t1:.2f}s")