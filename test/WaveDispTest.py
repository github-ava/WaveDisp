import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(__file__))

from test.TestForward import test_forward
from test.TestGeneration import test_generation
from test.TestTraining import test_training
from test.TestOptimization import optimization_test, add_optimization_test, add_multi_optimization_test

if __name__ == "__main__":
    # -----------------------------------
    # test_forward()
    # -----------------------------------
    # test_generation()
    # -----------------------------------
    # test_training()
    # -----------------------------------
    # optimization_test()
    # add_optimization_test()
    # add_multi_optimization_test()
    # -----------------------------------
    pass
