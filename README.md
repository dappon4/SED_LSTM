# SED_LSTM
## Introduction
This repository contains the implementation of the Real-time Sound Event Detection (SED) model using Long Short-Term Memory (LSTM) network. Report paper including implementation detail can be found [here](https://drive.google.com/file/d/1WVPxN2Ke6a7la7eBLBT2sjYa1qIzYQkR/view?usp=sharing)

## Dataset
The dataset used in this project is the [URBAN-SED](https://urbansed.weebly.com/). Version of the dataset used in this project is v2.0.0. This dataset is under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

after downloading the dataset, extract the files and place them in the `datasets/` folder. specify the path to the dataset in the `train.py` script arguments.  
default path assumes the dataset root is `../datasets/URBAN_SED/URBAN-SED_v2.0.0`
```bash
datasets
└── URBAN_SED
    └── URBAN-SED_v2.0.0
        ├── annotations
        ├── audio
        └── ...
SED_LSTM
├── train.py
└── ...
```

## Quick Start
### Create a virtual environment
It is recommended to use anaconda to create a virtual environment and install the required packages.
```bash
conda create -n sed_lstm python=3.12
conda activate sed_lstm
```
### Install required packages
```bash
pip install -r requirements.txt
```

### Clone the repository
```bash
git clone --depth 1 https://github.com/dappon4/SED_LSTM.git
```

## Training
### Train the model
```bash
python train.py
```
optional arguments:
- `--dataset_root`: path to the dataset (default: ../datasets/URBAN_SED/URBAN-SED_v2.0.0)
- `--batch_size`: batch size (default: 32)
- `--epochs`: number of epochs (default: 100)
- `--lr`: learning rate (default: 0.001)
- `--optimizer`: optimizer (default: adam)
- `--loss_fn`: loss function (default: focal)
- `--hidden_size`: hidden size of the LSTM (default: 256)
- `--load_all_data`: load all data at once (default: True) Note: set to False only if you do not have enough memory
- `--checkpoint_step`: save model every n epochs (default: 10)

### Check training progress
After each epoch, the model will log an image of validation data output in the `tmp/` folder. The image contains
- first row: input spectrogram
- second row: model output
- third row: model output after thresholding
- fourth row: ground truth label

### Tensorboard
To visualize the training progress, run the following command:
```bash
tensorboard --logdir runs
```
Then open a browser and go to `localhost:6006`.

### 
model weights will be saved at `model/` folder, under the folder with the starting time of the training.  
Additionally, you can find all the hyperparameters used in the training in the `summary.txt` file in the same folder.

## Adjusting post processing parameters
The post processing parameters can be adjusted in the `utility.py` `post_process` function.
- `threshold`: confidence threshold for the output of the model
- `min_duration`: minimum duration of the event in frames
- `max_gap`: maximum gap between the events in frames

We have provided a script to visualize and adjust the post processing parameters.

```bash
python adjust.py
```
Note: the purpose of this script is to visualize the effect of the post processing parameters. It sill NOT save the adjusted parameters.

## Evaluation
### run test script
```bash
python test.py --model <path to model>
```
example:
```bash
python test.py --model model/SED-Normal/model-best.pt
```
The script will generate a summary txt file at `test_output/` folder.
