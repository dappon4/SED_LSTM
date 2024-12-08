import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import numpy as np
import os
    
def display_imgs(mat):
    num, *_ = mat.shape
    
    fig, axs = plt.subplots(num, 1, figsize=(10, 8))
    for i in range(num):
        im_i = axs[i].imshow(mat[i,:,:], aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=1, cmap='inferno')
        fig.colorbar(im_i, ax=axs[i], label='Intensity')
    
    plt.show()

def save_output(spectrogram, output, label, epoch=0):
    # Select the first sequence in the batch
    spectrogram_data = spectrogram[0]
    output_data = output[0]
    label_data = label[0]
    # Apply sigmoid along the last dimension
    sigmoid_output = F.sigmoid(output_data)
    # Convert to NumPy array
    spectrogram_data = spectrogram_data.detach().cpu().numpy()
    sigmoid_output = sigmoid_output.detach().cpu().numpy()
    label_data = label_data.detach().cpu().numpy()
    
    # Create a figure with 2 subplots
    fig, axs = plt.subplots(5, 1, figsize=(10, 8))
    
    # Plot input spectrogram
    im0 = axs[0].imshow(spectrogram_data, aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=1, cmap='inferno')
    axs[0].set_xlabel('Time Steps')
    axs[0].set_ylabel('Classes')
    axs[0].set_title('Spectrogram Heatmap')
    fig.colorbar(im0, ax=axs[0], label='Intensity')
    
    # Plot output heatmap
    im1 = axs[1].imshow(sigmoid_output, aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=1, cmap='inferno')
    axs[1].set_xlabel('Time Steps')
    axs[1].set_ylabel('Classes')
    axs[1].set_title('Output Heatmap')
    fig.colorbar(im1, ax=axs[1], label='Probability')

    # confidence thershold 0.3
    im2 = axs[2].imshow(sigmoid_output > 0.3, aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=1, cmap='inferno')
    axs[2].set_xlabel('Time Steps')
    axs[2].set_ylabel('Classes')
    axs[2].set_title('threshold 0.3')
    fig.colorbar(im2, ax=axs[2], label='Probability')
    
    # confidence threshold 0.4
    
    im3 = axs[3].imshow(sigmoid_output > 0.4, aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=1, cmap='inferno')
    axs[3].set_xlabel('Time Steps')
    axs[3].set_ylabel('Classes')
    axs[3].set_title('threshold 0.4')
    fig.colorbar(im3, ax=axs[3], label='Probability')
    
    # Plot label heatmap
    im4 = axs[4].imshow(label_data, aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=1, cmap='inferno')
    axs[4].set_xlabel('Time Steps')
    axs[4].set_ylabel('Classes')
    axs[4].set_title('Label Heatmap')
    fig.colorbar(im4, ax=axs[4], label='Probability')
    
    # Save the figure
    plt.savefig(f"tmp/output_epoch_{epoch+1}.png")
    
    # return figure to log in tensorboard
    return fig

def save_model(model,time, name):
    os.makedirs(f"model/{time}", exist_ok=True)
    
    torch.save(model.state_dict(), f"model/{time}/{name}.pt")

def clear_tmp():
    os.makedirs('tmp', exist_ok=True)
    # remove all png in tmp folder
    for file in os.listdir('tmp'):
        if file.endswith('.png'):
            os.remove(f'tmp/{file}')

def post_process(output, threshold=0.4, min_duration=20, max_gap=3):
    """
    confidence_threshold: float, the threshold to consider a segment as positive
    min_duration: int, the minimum frames to consider a segment as positive
    max_gap: int, the maximum frames to consider as a gap
    """
    # output: (batch_size, num_classes, seq_len) numpy array
    batch_size, num_classes, seq_len = output.shape
    threshold_output = output > threshold
    
    for b in range(batch_size):
        for c in range(num_classes):
            # remove gaps
            
            binary_seq = threshold_output[b, c, :].astype(int)
            i = 0
            gap_start = None
            while i < seq_len:
                while i < seq_len and binary_seq[i] == 0:
                    i += 1
                segment_start = i
                
                if i >= seq_len:
                    break
                
                if gap_start is not None:
                    gap_len = segment_start - gap_start
                    if gap_len <= max_gap:
                        binary_seq[gap_start:segment_start] = 1
                
                while i < seq_len and binary_seq[i] == 1:
                    i += 1
                
                gap_start = i
            # remove short segments
            i = 0
            segment_start = None
            while i < seq_len:
                while i < seq_len and binary_seq[i] == 1:
                    i += 1
                segment_end = i-1
                if segment_start is not None:
                    segment_len = segment_end - segment_start + 1
                    if segment_len < min_duration:
                        binary_seq[segment_start:segment_end+1] = 0
                
                if i >= seq_len:
                    break
                
                while i < seq_len and binary_seq[i] == 0:
                    i += 1
                
                segment_start = i
            
            threshold_output[b, c, :] = binary_seq
            
    
    return threshold_output.astype(int)

def segment_to_time(prediction, actual_length=10):
    # prediction of shape (num_classes, seq_len)
    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                   'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
                   'siren', 'street_music', 'background']
    _, seq_len = prediction.shape
    
    lines = []
        
    for i in range(len(classes) - 1):
        class_pred = prediction[i]
        start = None
        end = None
        j = 0
        while j < seq_len:
            while j < seq_len and class_pred[j] == 0:
                j += 1
            if j >= seq_len:
                break
            start = j
            while j < seq_len and class_pred[j] == 1:
                j += 1
            end = j
            if start is not None:
                lines.append(f"{start/seq_len*actual_length:.5f},{end/seq_len*actual_length:.5f},{classes[i]}")
    return lines

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)