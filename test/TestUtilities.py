import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(__file__))
from src.ForwardProb import add_noise_to_cp_values, load_and_show_plots


def test_load_and_show_plots():
    load_and_show_plots(directory='../out/out_test/summary')


def test_add_noise_to_cp_values():
    add_noise_to_cp_values(['../data/data_test_0', '../data/data_test_1'], noise_percentage=0.5)


if __name__ == "__main__":
    test_load_and_show_plots()
    test_add_noise_to_cp_values()
