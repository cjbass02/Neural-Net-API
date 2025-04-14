import unittest
import torch
import numpy as np
import network  
from layers import Layer, Input, ReLU

class TestReLU(unittest.TestCase):
    def test_forward(self):
        # Create a dummy previous layer with mixed values.
        prev = Input(1, 2)
        prev_tensor = torch.tensor([[-1.0, 2.0]])
        prev.set(prev_tensor)
        # New ReLU signature: ReLU(rows, prev_output, cols=...)
        relu = ReLU(2, prev, cols=2)
        relu.forward()
        expected = torch.clamp(prev_tensor, min=0)

        print("\n\nTest ReLU Forwards: output =", relu.output, "expected =", expected)
        self.assertTrue(torch.allclose(relu.output, expected))



    def test_relu_backward(self):

        #expect 1 column.
        prev_layer = Input(3, 1)

        # Set a fake input to the relu 
        prev_output_val = torch.tensor([[-0.8], [1.2], [0.0]])
        prev_layer.set(prev_output_val)
        
        # Create a relu layer
        relu_layer = ReLU(3, prev_layer)
        
        #forward pass
        relu_layer.forward()
        
        # Incoming grad
        grad_val = torch.tensor([[2.0], [-1.0], [-2.0]])
        relu_layer.grad = grad_val
        
        prev_layer.clear_grad()
        
        # backwards pass
        relu_layer.backward()

        # From lab 8 derivations corrections
        expected_grad = torch.tensor([[0.0], [-1.0], [0.0]])
        
        # Actual Tests
        print("\n\n Test ReLU Backwards \n")
        print("prev_layer.grad:", prev_layer.grad, "\nexpected:", expected_grad)
        self.assertTrue(torch.allclose(prev_layer.grad, expected_grad),
                        f"Incorrect gradient for previous layer: expected {expected_grad}, got {prev_layer.grad}")

if __name__ == '__main__':
    unittest.main()