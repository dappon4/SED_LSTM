from torch import nn
import torch

class SED_LSTM(nn.Module):
    def __init__(self, mel_bins, lstm_input_size, hidden_size, num_classes, num_layers=1, bidirectional=False):
        super(SED_LSTM, self).__init__()
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, proj_size=num_classes, batch_first=True, num_layers=num_layers, bidirectional=bidirectional)
        
        self.input_proj = nn.Linear(mel_bins, lstm_input_size)
        self.softmax = nn.Softmax(dim=-1)
        
        self.bidirectional = bidirectional
    
    def forward(self, x):
        # input shape (batch, mel_bins, seq_len)
        x = x.permute(0, 2, 1) # (batch, seq_len, mel_bins)
        x = self.input_proj(x) # (batch, seq_len, lstm_input_size)
        x, _ = self.lstm(x) # (batch, seq_len, num_classes)
        # x = self.softmax(x) # not needed if we are using BCEWithLogitsLoss
        x = x.permute(0, 2, 1) # (batch, num_classes, seq_len)
        if self.bidirectional:
            x = x[:, x.size(1)//2:, :]  # only take the output from the second lstm block
        return x

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
    
    def forward(self, q, k, v):
        pass

class SED_Attention_LSTM(nn.Module):
    def __init__(self, mel_bins, lstm_input_size, hidden_size, num_classes, num_layers=1, **kwargs):
        super(SED_Attention_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(mel_bins, lstm_input_size)
        self.output_proj = nn.Linear(hidden_size, num_classes)
        
        self.linear_ii = nn.Linear(lstm_input_size, hidden_size)
        self.linear_hi = nn.Linear(hidden_size, hidden_size)
        self.linear_if = nn.Linear(lstm_input_size, hidden_size)
        self.linear_hf = nn.Linear(hidden_size, hidden_size)
        self.linear_ig = nn.Linear(lstm_input_size, hidden_size)
        self.linear_hg = nn.Linear(hidden_size, hidden_size)
        self.linear_io = nn.Linear(lstm_input_size, hidden_size)
        self.linear_ho = nn.Linear(hidden_size, hidden_size)
        self.linear_xo = nn.Linear(lstm_input_size, hidden_size)
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(1, hidden_size)
        self.attention_out_proj = nn.Linear(hidden_size, 1)
        
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch, seq_len, mel_bins)
        x = self.input_proj(x) # (batch, seq_len, lstm_input_size)
        x = self.lstm(x) # (batch, seq_len, num_classes)
        x = x.permute(0, 2, 1) # (batch, num_classes, seq_len)
        return x
    
    def lstm(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        output = []
        
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        output = []
        
        for t in range(seq_len):
            c, h = self._attention_lstm_block(c, h, x, t)
            output.append(self.output_proj(h))
        
        return torch.stack(output, dim=1)

    def _attention(self, c, o, x):
        q = self.q_proj(x).unsqueeze(2) # (batch, hidden_size, 1)
        k = self.k_proj(c).unsqueeze(2) # (batch, hidden_size, 1)
        v = self.v_proj(o.unsqueeze(2)) # (batch, hidden_size, hidden_size)
        
        attention = torch.matmul(q, k.transpose(1, 2)) / (self.hidden_size ** 0.5)
        attention = torch.softmax(attention, dim=-1) # (batch, hidden_size, hidden_size)
        out = torch.matmul(attention, v) # (batch, hidden_size, hidden_size)
        return self.attention_out_proj(out).squeeze()
        
    
    def _attention_lstm_block(self, c, h, x, t):
            x_t = x[:, t, :]
            i = torch.sigmoid(self.linear_ii(x_t) + self.linear_hi(h))
            f = torch.sigmoid(self.linear_if(x_t) + self.linear_hf(h))
            g = torch.tanh(self.linear_ig(x_t) + self.linear_hg(h))
            o = torch.sigmoid(self.linear_io(x_t) + self.linear_ho(h))
            c = f * c + i * g
            x_o = torch.sigmoid(self.linear_xo(x_t))
            h = self._attention(c, o, x_o)
            return c, h