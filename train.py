import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import URBAN_SED
from model import SED_LSTM, SED_Attention_LSTM, FocalLoss
from util import save_output, save_model, clear_tmp
import time
import argparse

def main(args):
    EPOCH = args.epochs
    LR = args.lr
    BATCH_SIZE = args.batch_size
    LOAD_ALL_DATA = args.load_all_data
    CHECKPOINT_STEP = args.checkpoint_step

    t1 = time.time()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataloader = DataLoader(URBAN_SED(args.dataset_root, split='train', preprocessed_dir='n_mels_64',load_all_data=LOAD_ALL_DATA, n_mels=64), batch_size=BATCH_SIZE, shuffle=True)
    validate_dataloader = DataLoader(URBAN_SED(args.dataset_root, split='validate', preprocessed_dir='n_mels_64', load_all_data=LOAD_ALL_DATA, n_mels=64), batch_size=BATCH_SIZE, shuffle=True)
    # test_dataloader = DataLoader(URBAN_SED('../datasets/URBAN_SED/URBAN-SED_v2.0.0', split='test', preprocessed_dir='base', load_all_data=LOAD_ALL_DATA), batch_size=BATCH_SIZE, shuffle=True)

    # model = SED_Attention_LSTM(mel_bins=64, lstm_input_size=128, hidden_size=256, num_classes=11, d_model=128).to('cuda')
    model = SED_LSTM(mel_bins=64, lstm_input_size=256, hidden_size=args.hidden_size, num_classes=11, num_layers=3, bidirectional=True, feature_extractor=args.feature_extractor).to(device)
    
    if args.optimzer == 'adam':
        optimzer = torch.optim.Adam(model.parameters(), lr=LR)
    elif args.optimzer == 'sgd':
        optimzer = torch.optim.SGD(model.parameters(), lr=LR)
    elif args.optimzer == 'adamw':
        optimzer = torch.optim.AdamW(model.parameters(), lr=LR)
    else:
        raise ValueError("Invalid optimzer type. Choose one of [adam, sgd, adamw]")

    if args.loss_fn == 'bce':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.loss_fn == 'focal':
        loss_fn = FocalLoss()

    folder_name = time.strftime("%Y%m%d-%H%M%S") + "-" + f"[{args.feature_extractor},{args.hidden_size},{LR},{args.optimzer},{args.loss_fn}]"
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
        
        if (epoch+1) % CHECKPOINT_STEP == 0 and (epoch+1) != EPOCH:
            save_model(model, folder_name, f"model-ckpt-{epoch+1}")
        
        if validate_loss < best_validate_loss:
            best_validate_loss = validate_loss
            best_epoch = epoch+1
            save_model(model, folder_name, f"model-best")

    # save the model
    save_model(model, folder_name, f"model")
    writer.close()  # Close the SummaryWriter
    with open(f"model/{folder_name}/summary.txt", 'w') as f:
        f.write(f"Parameters\n")
        f.write(f"\tHidden size: {args.hidden_size}\n")
        f.write(f"\tFeature extractor: {args.feature_extractor}\n")
        f.write(f"\tEpochs: {EPOCH}\n")
        f.write(f"\tLearning rate: {LR}\n")
        f.write(f"\tBatch size: {BATCH_SIZE}\n")
        f.write(f"\tOptimzer: {args.optimzer}\n")
        f.write(f"\tLoss function: {args.loss_fn}\n")
        f.write(f"\tLoad all data: {LOAD_ALL_DATA}\n")
        f.write(f"\tCheckpoint step: {CHECKPOINT_STEP}\n")
        f.write("\n")
        f.write(f"Training time: {time.time() - t1:.2f}s\n")
        f.write(f"Final validation loss: {validate_loss:.4f}\n")
        f.write(f"Best validation loss: {best_validate_loss:.4f} at epoch {best_epoch}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training hyperparameters')

    # Step 2: Define hyperparameter arguments
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size of the LSTM')
    parser.add_argument('--feature_extractor', type=str, default='normal', help='Feature extractor type, one of [normal, contextual, projection]')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--optimzer', type=str, default='adam', help='Optimzer type, one of [adam, sgd, adamw]')
    parser.add_argument('--loss_fn', type=str, default='focal', help='Loss function type, one of [focal, bce]')
    parser.add_argument('--load_all_data', type=bool, default=True, help='Load all data into memory')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='Checkpoint save step')
    parser.add_argument('--dataset_root', type=str, default='../datasets/URBAN_SED/URBAN-SED_v2.0.0', help='Root directory of the dataset')
    
    args = parser.parse_args()
    main(args)