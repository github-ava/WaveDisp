import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(__file__))
from src.ForwardProb import multi_train_network


def test_training():
    PrOpt = {  # Include combinations
        'max_epochs': [100],  # max epochs
        'optimizer': [0],  # 0: Adam, 1: Nadam
        'initializer': [0],  # 0: he_normal, 1:glorot_normal, 2: random_normal
        'activation': [0],  # 0: relu, 1: sigmoid, 2: LeakyReLU, 3: custom_leaky_relu
        'learning_rate': [1.e-4],
        'batch_size': [int(2 ** 7)],
        'loss': [2],  # 0: MSE, 1: MSLE, 2: MAE, 3: Huber
        'model': [0],  # 0: ANN , 0: CNN
        'in_dir': ['../data/data_test'],  # Synthetic training data path
    }
    PrOptRemoveList = (  # Exclude combinations
        {},
    )
    multi_train_network(PrOpt, PrOptRemoveList, base_out_dir='../out/out_test',
                        cuda_visible=-1, num_proc=1, tf_intra_op=0, tf_inter_op=0)


if __name__ == "__main__":
    test_training()
