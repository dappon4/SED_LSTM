from torch import nn
import torch

from preprocess import preprocess_audio

class SED_LSTM(nn.Module):
    def __init__(self, mel_bins, lstm_input_size, hidden_size, num_classes):
        super(SED_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, proj_size=num_classes, batch_first=True)
        
        self.input_proj = nn.Linear(mel_bins, lstm_input_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # input shape (batch, mel_bins, seq_len)
        x = x.permute(0, 2, 1) # (batch, seq_len, mel_bins)
        x = self.input_proj(x) # (batch, seq_len, lstm_input_size)
        x, _ = self.lstm(x) # (batch, seq_len, num_classes)
        # x = self.softmax(x) # not needed if we are using BCEWithLogitsLoss
        return x.permute(0, 2, 1) # (batch, num_classes, seq_len)