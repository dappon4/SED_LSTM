import os

training_commands = [
    "python train.py --hidden_size 256 --feature_extractor normal --epochs 100 --lr 0.0005 --optimzer adam --loss_fn focal",
    "python train.py --hidden_size 256 --feature_extractor normal --epochs 100 --lr 0.0005 --optimzer adam --loss_fn bce",
    "python train.py --hidden_size 256 --feature_extractor contextual --epochs 100 --lr 0.0005 --optimzer adam --loss_fn focal",
    "python train.py --hidden_size 256 --feature_extractor contextual --epochs 100 --lr 0.0005 --optimzer adam --loss_fn bce",
    "python train.py --hidden_size 256 --feature_extractor projection --epochs 100 --lr 0.0005 --optimzer adam --loss_fn focal",
    "python train.py --hidden_size 256 --feature_extractor projection --epochs 100 --lr 0.0005 --optimzer adam --loss_fn bce",
    "python train.py --hidden_size 256 --feature_extractor normal --epochs 100 --lr 0.0005 --optimzer adamw --loss_fn focal",
    "python train.py --hidden_size 256 --feature_extractor normal --epochs 100 --lr 0.0005 --optimzer sgd --loss_fn focal",
    "python train.py --hidden_size 512 --feature_extractor normal --epochs 100 --lr 0.0005 --optimzer adam --loss_fn focal",
    "python train.py --hidden_size 512 --feature_extractor normal --epochs 100 --lr 0.0005 --optimzer adamw --loss_fn focal",
    "python train.py --hidden_size 512 --feature_extractor contextual --epochs 100 --lr 0.0005 --optimzer adam --loss_fn focal",
    "python train.py --hidden_size 512 --feature_extractor contextual --epochs 100 --lr 0.0005 --optimzer adamw --loss_fn focal",
]

if __name__ == "__main__":
    for command in training_commands:
        os.system(command)