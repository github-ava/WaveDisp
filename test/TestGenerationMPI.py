import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(__file__))
from src.ForwardProb import generate_training_multi_layer_mpi


def test_generation_mpi():
    generate_training_multi_layer_mpi(n=3, num_sample=10, cs_range=[50., 650.], h_range=[1., 10.],
                                      h_fixed=-1., max_scale_factor=3., show_plots=False,
                                      out_dir='../data/data_test', num_proc=0)


if __name__ == "__main__":
    test_generation_mpi()
