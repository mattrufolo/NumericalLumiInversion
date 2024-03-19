import pytest
import tree_maker.NodeJob as nj

def test_method1():
    x = 1
    y = 2
    assert x + y == 3

#def test_sum_is_five():
#    assert nj.sum_is_five(1, 4) == True
    
def test_tag():
    a=nj()

