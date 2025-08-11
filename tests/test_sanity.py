from src.analysis import lift
import numpy as np

def test_lift_positive():
    assert lift([1,1,1],[2,2,2]) == 1.0

def test_lift_nan_safe():
    assert np.isnan(lift([0,0,0],[1,1,1]))