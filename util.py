import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import os

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