import os

training_commands = [
    "python train.py --hidden_size 512 --feature_extractor normal --num_layers 3 --epochs 60 --lr 0.0005 --optimzer adamw --loss_fn focal",
]

if __name__ == "__main__":
    for command in training_commands:
        os.system(command)