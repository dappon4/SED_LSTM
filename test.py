import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from dataset import URBAN_SED
from model import SED_LSTM
from tqdm import tqdm
from util import post_process, segment_to_time, display_imgs

import sed_eval
import dcase_util
import argparse

def main(args):
    # paht example: model/20241203-165317-SED-NormalFE/model-best.pt
    MODEL_PTH = args.model
    BATCH_SIZE = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SED_LSTM(mel_bins=64, lstm_input_size=256, hidden_size=256, num_classes=11, num_layers=3, bidirectional=True)
    model.load_state_dict(torch.load(MODEL_PTH, weights_only=True))

    test_dataloader = DataLoader(URBAN_SED('../datasets/URBAN_SED/URBAN-SED_v2.0.0', split='test', preprocessed_dir='n_mels_64', load_all_data=True, n_mels=64), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = tqdm(test_dataloader, desc="Generating annotation for test data", ncols=100)

    model = model.to(device)
    model.eval()

    save_path = "test_output/"+ MODEL_PTH.split("/")[-2]
    os.makedirs(f"{save_path}/annotations", exist_ok=True)

    # prepare evaluation csv

    ground_truth_prefix = "../datasets/URBAN_SED/URBAN-SED_v2.0.0/annotations/test/"
    pred_prefix = save_path + "/annotations/"

    index_df = pd.read_csv('../datasets/URBAN_SED/URBAN-SED_v2.0.0/test.csv')
    sed_eval_df = pd.DataFrame(columns=['ground_truth', 'prediction'])

    sed_eval_df['ground_truth'] = ground_truth_prefix + index_df['annotation']
    sed_eval_df['prediction'] = pred_prefix + index_df['annotation'].str.replace('.txt', '_pred.txt')

    sed_eval_df.to_csv(f"{save_path}/sed_eval.csv", index=False, header=False)

    # generate annotation files

    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                    'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
                    'siren', 'street_music', 'background']

    for i, (spectrogram, label) in enumerate(test_loader):
        with torch.no_grad():
            spectrogram = spectrogram.to(device)
            output = model(spectrogram)
            output = torch.sigmoid(output).detach().cpu().numpy()
        processed_output = post_process(output)
        
        # display_imgs(np.concat([processed_output[:1,:,:],label[:1,:,:],output[:1,:,:]], axis=0))
        
        for j in range(len(processed_output)):
            file_name = sed_eval_df.iloc[i*BATCH_SIZE+j]['prediction']
            lines = segment_to_time(processed_output[j,:,:]) # list of comma separated strings
            with open(f"{file_name}", 'w') as f:
                f.write("\n".join(lines))

    # Evaluate the results

    # Prepare file list from sed_eval_df
    file_list = sed_eval_df.to_dict(orient='records')

    data = []

    # Get used event labels
    all_data = dcase_util.containers.MetaDataContainer()
    for file_pair in file_list:
        reference_event_list = sed_eval.io.load_event_list(
            filename=file_pair['ground_truth']
        )
        estimated_event_list = sed_eval.io.load_event_list(
            filename=file_pair['prediction']
        )

        data.append({'reference_event_list': reference_event_list,
                    'estimated_event_list': estimated_event_list})

        all_data += reference_event_list

    event_labels = all_data.unique_event_labels

    # Start evaluating

    # Create metrics classes, define parameters
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=event_labels,
        time_resolution=1.0
    )

    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=event_labels,
        t_collar=0.200
    )

    # Go through files
    for file_pair in data:
        segment_based_metrics.evaluate(
            reference_event_list=file_pair['reference_event_list'],
            estimated_event_list=file_pair['estimated_event_list']
        )

        event_based_metrics.evaluate(
            reference_event_list=file_pair['reference_event_list'],
            estimated_event_list=file_pair['estimated_event_list']
        )

    with open(f"{save_path}/result_metrics.txt", 'w') as f:
        f.write(str(segment_based_metrics))
        f.write(str(event_based_metrics))

    print(f"result saved at {save_path}/result_metrics.txt")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model path')
    parser.add_argument('--model', type=str, help='path to the model')
    args = parser.parse_args()
    main(args)