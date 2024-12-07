from model import SED_LSTM
from dataset import URBAN_SED
import torch
import matplotlib.pyplot as plt
import numpy as np
from util import post_process
import os
import librosa as lr

os.makedirs('tmp/resource', exist_ok=True)

model_path = 'model/20241206-211210-[normal,512,0.0005,adamw,focal,False]/model-ckpt-50.pt'
model = SED_LSTM(mel_bins=64, lstm_input_size=256, hidden_size=512, num_classes=11, num_layers=3, bidirectional=False, feature_extractor='normal', num_frames=5)
model.load_state_dict(torch.load(model_path, weights_only=True))

spectrogram_idx = 6 # 5

train_dataset = URBAN_SED('../datasets/URBAN_SED/URBAN-SED_v2.0.0', split='train', preprocessed_dir='n_mels_64', load_all_data=True, n_mels=64)
spectrogram, label = train_dataset.__getitem__(spectrogram_idx)
spectrogram = spectrogram[:,:430]
label = label.detach().cpu().numpy()

with torch.no_grad():
    model_output = torch.sigmoid(model(spectrogram.unsqueeze(0))).detach().cpu().numpy()
processed_output = post_process(model_output)

# create plot for post process section

classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                   'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
                   'siren', 'street_music', 'background']

fig, ax = plt.subplots(4, 1, figsize=(12, 8))

im0 = ax[0].imshow(model_output[0], aspect='auto', origin='lower', cmap='inferno', vmin=0, vmax=1)
ax[0].set_title('Model Output')
ax[0].set_yticks(range(len(classes)))
ax[0].set_yticklabels(classes)

im1 = ax[1].imshow(model_output[0] > 0.4, aspect='auto', origin='lower', cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
ax[1].set_title('Thresholded Model Output')
ax[1].set_yticks(range(len(classes)))
ax[1].set_yticklabels(classes)

im2 = ax[2].imshow(processed_output[0], aspect='auto', origin='lower', cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
ax[2].set_title('Post Processed Model Output')
ax[2].set_yticks(range(len(classes)))
ax[2].set_yticklabels(classes)

im3 = ax[3].imshow(label, aspect='auto', origin='lower', cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
ax[3].set_title('Ground Truth Label')
ax[3].set_xlabel('Time Steps')
ax[3].set_yticks(range(len(classes)-1))
ax[3].set_yticklabels(classes[:-1])

plt.tight_layout()
plt.savefig('tmp/resource/all_in_one.png')

# create plot for model architecture section
# save spectrogram image without white margins
spectrogram = spectrogram.detach().cpu().numpy()
S_db = lr.power_to_db(spectrogram, ref=np.max)
plt.imshow(S_db, aspect='auto', origin='lower', cmap='inferno')
plt.axis('off')
plt.savefig('tmp/resource/spectrogram.png', bbox_inches='tight', pad_inches=0)

# save model output image without white margins
plt.imshow(model_output[0], aspect='auto', origin='lower', cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
plt.axis('off')
plt.savefig('tmp/resource/model_output.png', bbox_inches='tight', pad_inches=0)

# save post processed output image without white margins
plt.imshow(processed_output[0], aspect='auto', origin='lower', cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
plt.axis('off')
plt.savefig('tmp/resource/post_processed_output.png', bbox_inches='tight', pad_inches=0)

# save ground truth label image without white margins
plt.imshow(label, aspect='auto', origin='lower', cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
plt.axis('off')
plt.savefig('tmp/resource/ground_truth_label.png', bbox_inches='tight', pad_inches=0)