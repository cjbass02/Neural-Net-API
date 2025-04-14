import unittest
import torch
import numpy as np
import network
from layers import Layer, Input

class TestAccumulate(unittest.TestCase):
    def test_accumulate_grad(self):
        # Create a fake weight layer
        param_layer = Input(3, 1)
        # set the weights initial grad
        initial_grad = torch.tensor([[0.01], [0.2], [-0.03]])
        param_layer.grad = initial_grad
        
        
        # Accumulate a couple of gradients
        grad1 = torch.tensor([[0.01], [0.001], [0.2]])
        
        param_layer.accumulate_grad(grad1)
        
        # The expected accumulated gradient is simply the sum of grad1 and grad2
        expected_grad = torch.tensor([[0.02], [0.201], [0.17]])


        print("Expected accu grad: " + str(expected_grad))
        print("Actual calculated grad: " + str(param_layer.grad))
        
        # Check that the layer's grad attribute matches the expected value
        self.assertTrue(torch.allclose(param_layer.grad, expected_grad, atol=1e-5),
                        f"Accumulated gradient incorrect: expected \n{expected_grad}, got \n{param_layer.grad}")

if __name__ == '__main__':
    unittest.main()