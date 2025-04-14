import unittest
import torch
import numpy as np
import network
from layers import Layer, Input

class TestInput(unittest.TestCase):
    def test_set_valid(self):
        rows, cols = 2, 3
        inp = Input(rows, cols)
        valid_tensor = torch.tensor([[1.0, 2.0, 3.0],
                                     [4.0, 5.0, 6.0]])
        inp.set(valid_tensor)
        print("TestInput (set valid): output =", inp.output, "expected =", valid_tensor)
        self.assertTrue(torch.equal(inp.output, valid_tensor))
        
    def test_set_invalid(self):
        rows, cols = 2, 3
        inp = Input(rows, cols)
        invalid_tensor = torch.tensor([[1.0, 2.0],
                                       [3.0, 4.0]])
        with self.assertRaises(ValueError):
            print("TestInput (set invalid): expecting ValueError for tensor shape", invalid_tensor.shape)
            inp.set(invalid_tensor)
            
    def test_forward_noop(self):
        # The forward for Input does nothing, so output is the same
        rows, cols = 2, 3
        inp = Input(rows, cols)
        valid_tensor = torch.tensor([[1.0, 2.0, 3.0],
                                     [4.0, 5.0, 6.0]])
        inp.set(valid_tensor)
        inp.forward()
        print("TestInput (forward noop): output =", inp.output, "expected =", valid_tensor)
        self.assertTrue(torch.equal(inp.output, valid_tensor))

if __name__ == '__main__':
    unittest.main()