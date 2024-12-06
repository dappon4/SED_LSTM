from util import *
from dataset import URBAN_SED
from model import SED_LSTM
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
import random

validation_data = URBAN_SED('../datasets/URBAN_SED/URBAN-SED_v2.0.0', split='validate', preprocessed_dir='n_mels_64', load_all_data=True, n_mels=64)

MODEL_PTH = "model/20241203-165317-SED-NormalFE/model-best.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SED_LSTM(mel_bins=64, lstm_input_size=256, hidden_size=256, num_classes=11, num_layers=3, bidirectional=True)
model.load_state_dict(torch.load(MODEL_PTH, weights_only=True))
model = model.to(device)
model.eval()

def interactive_plot(model, threshold=0.4, min_duration=20, max_gap=3, random_samples = 5):
    
    fig, axs = plt.subplots(4, random_samples, figsize=(15, 12))
    ax_threshold = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_max_gap = plt.axes([0.25, 0.10, 0.65, 0.03])
    ax_min_duration = plt.axes([0.25, 0.05, 0.65, 0.03])
    
    slider_threshold = Slider(ax_threshold, 'Threshold', 0.0, 1.0, valinit=threshold, valstep=0.05)
    slider_max_gap = Slider(ax_max_gap, 'Max Gap', 0, 10, valinit=max_gap, valstep=1)
    slider_min_duration = Slider(ax_min_duration, 'Min Duration', 1, 40, valinit=min_duration, valstep=1)
    
    def get_new_samples():
        
        global output
        
        spectrograms = []
        labels = []
        
        random_idices = random.sample(range(len(validation_data)), random_samples)
        for i in random_idices:
            spec, label = validation_data.__getitem__(i)
            spectrograms.append(spec.unsqueeze(0))
            labels.append(label)
            
        with torch.no_grad():
            input_spectrogram = torch.cat(spectrograms).to(device)
            output = torch.sigmoid(model(input_spectrogram)).cpu().numpy()

        post_processed = post_process(output, slider_threshold.val, slider_min_duration.val, slider_max_gap.val)
        
        for idx in range(random_samples):
            axs[0, idx].imshow(output[idx], aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=1, cmap='inferno')
            axs[0, idx].set_title(f"Sample {random_idices[idx]}")
            
            axs[1, idx].imshow(output[idx] > threshold, aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=1, cmap='inferno')
            
            axs[2, idx].imshow(post_processed[idx], aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=1, cmap='inferno')
            
            axs[3, idx].imshow(labels[idx], aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=1, cmap='inferno')
    
    
    
    def update(val):
        threshold = slider_threshold.val
        max_gap = int(slider_max_gap.val)
        min_duration = int(slider_min_duration.val)
        
        post_processed = post_process(output, threshold=threshold, min_duration=min_duration, max_gap=max_gap)
        
        for idx in range(random_samples):
            axs[1, idx].imshow(output[idx] > threshold, aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=1, cmap='inferno')
            axs[2, idx].imshow(post_processed[idx], aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=1, cmap='inferno')
        
        fig.canvas.draw_idle()
    
    get_new_samples()
    
    plt.subplots_adjust(left=0.25, bottom=0.25)
    
    # Add button for loading new random samples
    ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(ax_button, 'New Samples')
    button.on_clicked(lambda event: get_new_samples())
    
    slider_threshold.on_changed(update)
    slider_max_gap.on_changed(update)
    slider_min_duration.on_changed(update)
    
    plt.show()


# Example usage
interactive_plot(model)



