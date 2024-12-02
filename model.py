from torch import nn, Tensor
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
        x = torch.relu(x)
        x, _ = self.lstm(x) # (batch, seq_len, num_classes)
        # x = self.softmax(x) # not needed if we are using BCEWithLogitsLoss
        x = x.permute(0, 2, 1) # (batch, num_classes, seq_len)
        if self.bidirectional:
            x = x[:, x.size(1)//2:, :]  # only take the output from the second lstm block
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, mel_bins, lstm_input_size):
        super().__init__()
        
        self.mel_bins = mel_bins
        
        self.initial_proj = nn.Linear(mel_bins, 4*lstm_input_size)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch, seq_len, mel_bins)
        batch_size, seq_len = x.size(0), x.size(1)
        x = self.initial_proj(x) # (batch, seq_len, 4*lstm_input_size)
        x = torch.relu(x)
        x = x.view(-1, 1, x.size(-1)) # (batch*seq_len, 1, 4*lstm_input_size)
        
        x = self.conv1(x) # (batch*seq_len, 16, 4*lstm_input_size)
        x = torch.relu(x)
        x = self.pool(x) # (batch*seq_len, 16, 2*lstm_input_size)
        x = self.conv2(x) # (batch*seq_len, 32, 2*lstm_input_size)
        x = torch.relu(x)
        x = self.pool(x) # (batch*seq_len, 32, lstm_input_size)
        x = self.conv3(x) # (batch*seq_len, 1, lstm_input_size)
        x = x.view(batch_size, seq_len, -1) # (batch, seq_len, lstm_input_size)
        return x

class SED_Attention_LSTM(nn.Module):
    def __init__(self, mel_bins, lstm_input_size, hidden_size, num_classes, d_model=256):
        super(SED_Attention_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(mel_bins, lstm_input_size)
        self.output_proj = nn.Linear(hidden_size, num_classes)
        self.d_model = d_model
        
        self.feature_extractor = FeatureExtractor(mel_bins, lstm_input_size)
        
        self.linear_ii = nn.Linear(lstm_input_size, hidden_size)
        self.linear_hi = nn.Linear(hidden_size, hidden_size)
        self.linear_if = nn.Linear(lstm_input_size, hidden_size)
        self.linear_hf = nn.Linear(hidden_size, hidden_size)
        self.linear_ig = nn.Linear(lstm_input_size, hidden_size)
        self.linear_hg = nn.Linear(hidden_size, hidden_size)
        self.linear_io = nn.Linear(lstm_input_size, hidden_size)
        self.linear_ho = nn.Linear(hidden_size, hidden_size)
        self.linear_xo = nn.Linear(lstm_input_size, hidden_size)
        
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)
        self.q_proj = nn.Linear(hidden_size, hidden_size*d_model)
        self.k_proj = nn.Linear(hidden_size, hidden_size*d_model)
        self.v_proj = nn.Linear(hidden_size, hidden_size*d_model)
        self.attention_out_proj = nn.Linear(d_model, 1)
        
        
    def forward(self, x):
        x = self.feature_extractor(x)
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
        q = self.q_proj(x).view(-1, self.hidden_size, self.d_model) # (batch, hidden_size, d_model)
        k = self.k_proj(c).view(-1, self.hidden_size, self.d_model) # (batch, hidden_size, d_model)
        v = self.v_proj(o).view(-1, self.hidden_size, self.d_model) # (batch, hidden_size, hidden_size)
        
        # out = self.attention(q, k, v)[0] # (batch, hidden_size, hidden_size)
        
        attention = torch.matmul(q, k.transpose(1, 2)) / (self.d_model ** 0.5)
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        epsilon = 1e-6  # Small constant to prevent log(0)
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
        pt = target * pred + (1 - target) * (1 - pred)
        pt = torch.clamp(pt, epsilon, 1.0 - epsilon)
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss