import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(__file__))
from src.ForwardProb import multi_train_network


def test_training():
    PrOpt = {  # Include combinations
        'in_dir': ['../data/data_test'],
        'max_epochs': [200],
        # 'optimizer': [0, 1],  # 0: Adam, 1: Nadam
        # 'initializer': [0, 1, 2],  # 0: he_normal, 1:glorot_normal, 2: random_normal
        # 'activation': [0, 1, 2, 3],  # 0: relu, 1: sigmoid, 2: LeakyReLU, 3: custom_leaky_relu
        # 'learning_rate': [1.e-2, 1.e-3],
        # 'batch_size': [int(2 ** 6), int(2 ** 7)],
        # 'model': [0],
        # 'h_fixed': [-1.],
        # 'optimization': [False],
    }
    PrOptRemoveList = (  # Exclude combinations
        {},
    )
    multi_train_network(PrOpt, PrOptRemoveList, base_out_dir='../out/out_train',
                        cuda_visible=-1, num_proc=0, tf_intra_op=0, tf_inter_op=0)


if __name__ == "__main__":
    test_training()
