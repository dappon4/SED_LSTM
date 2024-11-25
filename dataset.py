import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import librosa as lr
from tqdm import tqdm

class URBAN_SED(Dataset):
    def __init__(self, root, transform=None, target_transform=None, split='train', load_all_data=False, **kwargs):
        assert split in ['train', 'validate', 'test'], 'Invalid split'
        self.root = root
        self.split = split
        self.data = self.get_df()
        self.transform = transform
        self.target_transform = target_transform
        self.kwargs = kwargs
        self.min_seqlen = self.get_min_seqlen()
        self.load_all_data = load_all_data
        
        if load_all_data:
            self.all_data, self.all_labels = self.get_all_data()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.load_all_data:
            spectrogram = self.all_data[idx]
            label = self.all_labels[idx]
        else:
            audio_path, annotation_path, duration = self.data.iloc[idx]
            audio, sr = lr.load(f"{self.root}/audio/{self.split}/{audio_path}")
            hop_length = self.kwargs.get('hop_length', 512)
            n_fft = self.kwargs.get('n_fft', 2048)
            
            spectrogram = torch.Tensor(lr.feature.melspectrogram(y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft))[:, :self.min_seqlen]
            label = self.get_label(annotation_path, spectrogram.shape)

        if self.transform:
            spectrogram = self.transform(spectrogram)
        if self.target_transform:
            label = self.target_transform(label)
        
        return spectrogram, label
    
    def get_df(self):
        # index files if csv not found at root directory
        if not os.path.exists(f"{self.root}/{self.split}.csv"):
            self.index_files()
            
        df = pd.read_csv(f"{self.root}/{self.split}.csv")
        return df
        
    def get_label(self, annotation_path, shape):
        # Define the list of classes
        classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                   'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
                   'siren', 'street_music']
        
        # Initialize the label tensor
        # The last class is for the background noise
        label = torch.zeros(len(classes)+1, shape[1])
        
        # Load the annotation file
        df = pd.read_csv(f"{self.root}/annotations/{self.split}/{annotation_path}", 
                         sep='\t', header=None, names=['start', 'end', 'label'])
        
        # Get hop_length and sample rate
        hop_length = self.kwargs.get('hop_length', 512)
        sr = 22050  # Default sample rate, adjust if necessary
        
        # Map annotations to spectrogram frames
        for _, row in df.iterrows():
            if row['label'] in classes:
                class_idx = classes.index(row['label'])
                start_frame = int(row['start'] * sr / hop_length)
                end_frame = int(row['end'] * sr / hop_length)
                label[class_idx, start_frame:end_frame] = 1
        
        # The last class is for the background noise
        label[-1] = 1 - label[:-1].max(dim=0)[0]
        
        return label

    def index_files(self):
        assert os.path.exists(self.root), 'Root directory not found'
        assert os.path.isdir(f"{self.root}/audio/{self.split}"), 'Audio directory not found'
        
        print(f"Indexing {self.split} files")
        
        audio_files = os.listdir(f"{self.root}/audio/{self.split}")
        
        df = pd.DataFrame([audio_files]).T
        # remove row starting with "."
        df = df[~df[0].str.startswith('.')]
        
        # add column of annotations changing wav to txt
        df[1] = df[0].str.replace('.wav', '.txt')
        
        # get the duration of the audio files of df[0]
        durations = []
        for audio in df[0]:
            audio, sr = lr.load(f"{self.root}/audio/{self.split}/{audio}")
            durations.append(lr.get_duration(y=audio, sr=sr))
        
        df[2] = durations
        
        # test if annotation files exist
        df.columns = ['audio', 'annotation', "duration"]
        
        # test if annotation files exist
        for idx, row in df.iterrows():
            if not os.path.exists(f"{self.root}/annotations/{self.split}/{row['annotation']}"):
                print(f"Annotation file {row['annotation']} not found")
                df.drop(idx, inplace=True)
        
        df.to_csv(f"{self.root}/{self.split}.csv", index=False)
    
    def get_min_seqlen(self):
        min_duration = self.data['duration'].min()
        return int(min_duration * 22050 / self.kwargs.get('hop_length', 512)) + 1
    
    def get_all_data(self):
        all_data = []
        all_labels = []
        for idx in tqdm(range(len(self.data)), desc=f"Loading {self.split} data"):
            audio_path, annotation_path, duration = self.data.iloc[idx]
            audio, sr = lr.load(f"{self.root}/audio/{self.split}/{audio_path}")
            hop_length = self.kwargs.get('hop_length', 512)
            n_fft = self.kwargs.get('n_fft', 2048)

            spectrogram = torch.Tensor(lr.feature.melspectrogram(y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft))[:, :self.min_seqlen]
            label = self.get_label(annotation_path, spectrogram.shape)
            all_data.append(spectrogram)
            all_labels.append(label)
        
        return all_data, all_labels
    
if __name__ == '__main__':
    data = URBAN_SED('../datasets/URBAN_SED/URBAN-SED_v2.0.0', split='train')
    
    data.__getitem__(0)
    data.__getitem__(3500)