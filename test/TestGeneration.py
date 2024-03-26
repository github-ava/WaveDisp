import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(__file__))
from src.ForwardProb import generate_training_multi_layer


def test_generation():
    # Change the working directory to the directory where the Python file is located

    generate_training_multi_layer(n_min=3, n_max=3, num_sample=10, cs_range=[50., 650.], h_range=[1., 10.],
                                  h_fixed=-1., max_scale_factor=3., show_plots=False,
                                  out_dir='../data/data_test', num_proc=0)


if __name__ == "__main__":
    test_generation()
