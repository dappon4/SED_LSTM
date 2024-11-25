import librosa as lr
import numpy as np
import matplotlib.pyplot as plt

def display_waveform(spectrogram):
    # visualize waveform
    lr.display.specshow(lr.power_to_db(spectrogram, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

def preprocess_audio(audio_path, sr=16000):
    sf, sr = lr.load(audio_path)
    spectrogram = lr.feature.melspectrogram(y=sf, sr=sr)
    print(spectrogram.shape)
    #display_waveform(spectrogram)

    return spectrogram

if __name__ == '__main__':
    audiopath = '../datasets/URBAN_SED/URBAN-SED_v2.0.0/audio/train/soundscape_train_bimodal0.wav'
    preprocess_audio(audiopath)