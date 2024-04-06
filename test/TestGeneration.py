import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(__file__))
from src.ForwardProb import generate_training_multi_layer


def test_generation():
    # h_fixed = -1. means variable thickness for each layer
    #         = 1.  means fixed thickness 1. for all layers
    #         = 0.  means variable thickness but same for all layers
    generate_training_multi_layer(n=[3], num_sample=[10], cs_range=[50., 650.], h_range=[1., 10.],
                                  h_fixed=-1., max_scale_factor=3., show_plots=False,
                                  out_dir='../data/data_test', num_proc=0)
    # generate_training_multi_layer(n=[10], num_sample=[10], cs_range=[50., 650.], h_range=[2., 2.],
    #                               h_fixed=2., max_scale_factor=3., show_plots=False,
    #                               out_dir='../data/data_01', num_proc=0)
    # generate_training_multi_layer(n=[10], num_sample=[10], cs_range=[50., 650.], h_range=[1.5, 2.5],
    #                               h_fixed=0., max_scale_factor=3., show_plots=False,
    #                               out_dir='../data/data_test', num_proc=0)


if __name__ == "__main__":
    test_generation()
