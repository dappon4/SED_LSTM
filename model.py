from torch import nn, Tensor
import torch


class SED_LSTM(nn.Module):
    def __init__(self, mel_bins, lstm_input_size, hidden_size, num_classes, num_layers=3, bidirectional=False, feature_extractor="normal", **kwargs):
        super(SED_LSTM, self).__init__()
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=bidirectional)
        
        if feature_extractor == "normal":
            self.feature_extractor = FeatureExtractor(mel_bins, lstm_input_size)
        elif feature_extractor == "contextual":
            num_frames = kwargs.get('num_frames', 10)
            self.feature_extractor = ContextualFeatureExtractor(mel_bins, lstm_input_size, num_frames=num_frames)
        elif feature_extractor == "projection":
            self.feature_extractor = FeatureProjection(mel_bins, lstm_input_size)
        elif feature_extractor == "combined":
            num_frames = kwargs.get('num_frames', 8)
            self.feature_extractor = CombinedFeatureExtractor(mel_bins, lstm_input_size, num_frames)
        
        self.input_proj = nn.Linear(mel_bins, lstm_input_size)
        self.ff_1 = nn.Linear(hidden_size, hidden_size // 2)
        self.ff_2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.ff_3 = nn.Linear(hidden_size // 4, num_classes)
        

    
    def forward(self, x):
        # input shape (batch, mel_bins, seq_len)
        x = self.feature_extractor(x)
        x, _ = self.lstm(x) # (batch, seq_len, hidden_size)

        x = self.ff_1(x)
        x = torch.relu(x)
        x = self.ff_2(x)
        x = torch.relu(x)
        x = self.ff_3(x)
        
        return x.permute(0, 2, 1) # (batch, num_classes, seq_len)

class FeatureProjection(nn.Module):
    def __init__(self, mel_bins, lstm_input_size):
        super().__init__()
        self.proj = nn.Linear(mel_bins, lstm_input_size)
    
    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch, seq_len, mel_bins)
        return self.proj(x)

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
    
class ContextualFeatureExtractor(nn.Module):
    def __init__(self, mel_bins, lstm_input_size, num_frames):
        super().__init__()
        self.num_frames = num_frames
        self.lstm_input_size = lstm_input_size
        
        self.initial_proj = nn.Linear(mel_bins, 4*lstm_input_size)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(num_frames, 3), padding=(0, 1))
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 1, kernel_size=3, padding=1)
        
        self.pool2d = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool1d = nn.MaxPool1d(kernel_size=2)
    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch, seq_len, mel_bins)
        batch_size, seq_len = x.size(0), x.size(1)
        x = self.initial_proj(x) # (batch, seq_len, 4*lstm_input_size)
        pad_matrix = torch.zeros(x.size(0), self.num_frames-1, 4*self.lstm_input_size).to(x.device)
        x = torch.cat([pad_matrix, x], dim=1) # (batch, seq_len+num_frames-1, 4*lstm_input_size)
        x = x.unsqueeze(1) # (batch, 1, seq_len+num_frames-1, 4*lstm_input_size)
        
        x = self.conv1(x) # (batch, 16, seq_len, 4*lstm_input_size)
        x = torch.relu(x)
        x = self.pool2d(x) # (batch, 16, seq_len, 2*lstm_input_size)
        
        x = x.permute(0, 2, 1, 3) # (batch, seq_len, 16, 2*lstm_input_size)
        x = x.reshape(batch_size*seq_len, 16, 2*self.lstm_input_size) # (batch*seq_len, 16, 2*lstm_input_size)
        x = self.conv2(x) # (batch*seq_len, 32, 2*lstm_input_size)
        x = torch.relu(x)
        x = self.pool1d(x) # (batch*seq_len, 32, lstm_input_size)
        
        x = self.conv3(x) # (batch*seq_len, 1, lstm_input_size)
        x = x.view(batch_size, seq_len, -1) # (batch, seq_len, lstm_input_size)
        return x

class ContextualFeatureExtractor_v2(nn.Module):
    def __init__(self, mel_bins, lstm_input_size, num_frames):
        super().__init__()
        self.num_frames = num_frames
        self.lstm_input_size = lstm_input_size
        
        self.initial_proj = nn.Linear(mel_bins, lstm_input_size)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.pool2d = nn.MaxPool2d(kernel_size=2)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.ff_1 = nn.Linear(32*(num_frames//4)*(lstm_input_size//4), 8*num_frames//4*lstm_input_size//4)
        self.ff_2 = nn.Linear(8*num_frames//4*lstm_input_size//4, lstm_input_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size, seq_len = x.size(0), x.size(1)
        x = self.initial_proj(x) # (batch, seq_len, lstm_input_size)
        x = torch.relu(x)
        pad_matrix = torch.zeros(x.size(0), self.num_frames-1, self.lstm_input_size).to(x.device)
        x = torch.cat([pad_matrix, x], dim=1) # (batch, seq_len+num_frames-1, lstm_input_size)
        x = x.unfold(dimension=1, size=self.num_frames, step=1) # (batch, seq_len, num_frames, lstm_input_size)
        x = x.reshape(batch_size * seq_len, 1, self.num_frames, self.lstm_input_size)
        
        x = self.conv1(x) # (batch * seq_len, 16, num_frames, lstm_input_size)
        x = torch.relu(x)
        x = self.pool2d(x) # (batch * seq_len, 16, num_frames//2, lstm_input_size//2)
        x = self.bn1(x)
        
        x = self.conv2(x) # (batch * seq_len, 32, num_frames//2, lstm_input_size//2)
        x = torch.relu(x)
        x = self.pool2d(x) # (batch * seq_len, 32, num_frames//4, lstm_input_size//4)
        x = self.bn2(x)
        
        x = x.view(batch_size, seq_len, -1) # (batch * seq_len, 32*num_frames//4*lstm_input_size//4)
        x = self.ff_1(x) # (batch * seq_len, 8*num_frames//4*lstm_input_size//4)
        x = torch.relu(x)
        x = self.ff_2(x) # (batch * seq_len, lstm_input_size)
        x = torch.relu(x)
        return x.view(batch_size, seq_len, -1) # (batch, seq_len, lstm_input_size)
        
class CombinedFeatureExtractor(nn.Module):
    def __init__(self, mel_bins, lstm_input_size, num_frames):
        super().__init__()
        self.normal = FeatureExtractor(mel_bins, lstm_input_size)
        self.contextual = ContextualFeatureExtractor_v2(mel_bins, lstm_input_size, num_frames)
        
        
    def forward(self, x):
        merged = 0.5 * (self.normal(x) + self.contextual(x))
        return torch.relu(merged)
    
class SED_Attention_LSTM(nn.Module):
    def __init__(self, mel_bins, lstm_input_size, hidden_size, num_classes, d_model=256):
        super(SED_Attention_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(mel_bins, lstm_input_size)
        self.output_proj = nn.Linear(hidden_size, num_classes)
        self.d_model = d_model
        
        self.feature_extractor = ContextualFeatureExtractor(mel_bins, lstm_input_size)
        
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
        if torch.isnan(pt).any():
            print('NAN in pt')
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss