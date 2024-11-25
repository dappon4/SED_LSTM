import matplotlib.pyplot as plt
import torch.nn.functional as F

def save_output(spectrogram, output, label, epoch=0):
    # Select the first sequence in the batch
    spectrogram_data = spectrogram[0]
    output_data = output[0]
    label_data = label[0]
    # Apply sigmoid along the last dimension
    sigmoid_spectrogram = F.sigmoid(spectrogram_data)
    sigmoid_output = F.sigmoid(output_data)
    sigmoid_label = F.sigmoid(label_data)
    # Convert to NumPy array
    sigmoid_spectrogram = sigmoid_spectrogram.detach().cpu().numpy()
    sigmoid_output = sigmoid_output.detach().cpu().numpy()
    sigmoid_label = sigmoid_label.detach().cpu().numpy()
    
    # Create a figure with 2 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plot input spectrogram
    im0 = axs[0].imshow(sigmoid_spectrogram, aspect='auto', origin='lower', interpolation='nearest')
    axs[0].set_xlabel('Time Steps')
    axs[0].set_ylabel('Classes')
    axs[0].set_title('Spectrogram Heatmap')
    fig.colorbar(im0, ax=axs[0], label='Probability')
    
    # Plot output heatmap
    im1 = axs[1].imshow(sigmoid_output, aspect='auto', origin='lower', interpolation='nearest')
    axs[1].set_xlabel('Time Steps')
    axs[1].set_ylabel('Classes')
    axs[1].set_title('Output Heatmap')
    fig.colorbar(im1, ax=axs[1], label='Probability')

    # Plot label heatmap
    im2 = axs[2].imshow(sigmoid_label, aspect='auto', origin='lower', interpolation='nearest')
    axs[2].set_xlabel('Time Steps')
    axs[2].set_ylabel('Classes')
    axs[2].set_title('Label Heatmap')
    fig.colorbar(im2, ax=axs[2], label='Probability')
    
    # Save the figure
    plt.savefig(f"tmp/output_epoch_{epoch}.png")
    
    # return figure to log in tensorboard
    return fig
    