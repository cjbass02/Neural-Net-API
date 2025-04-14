from unittest import TestCase
from layers import Layer, Input, Sum
import numpy as np
import torch
import unittest

# With this block, we don't need to set device=DEVICE for every tensor.
# But you will still need to avoid accidentally getting int types instead of floating-point types.
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
     torch.set_default_device(0)
     print("Running on the GPU")
else:
     print("Running on the CPU")

class TestSum(unittest.TestCase):
     def test_forward(self):
          # Create two input layers 
          prev1 = Input(1, 2)
          prev2 = Input(1, 2)
          
          tensor1 = torch.tensor([[1.0, 2.0]])
          prev1.set(tensor1)
          tensor2 = torch.tensor([[3.0, 4.0]])
          prev2.set(tensor2)
          
          # Create a Sum layer 
          sum_layer = Sum(prev1, prev2, rows=1, cols=2)
          sum_layer.forward()
          
          # The expected output
          expected = torch.tensor([[4.0, 6.0]])
          
          print("TestSum: output =", sum_layer.output, "expected =", expected)
          self.assertTrue(torch.allclose(sum_layer.output, expected))
     

     def test_sum_backward(self):
          # left and right layers
          left_layer = Input(3, 1)
          right_layer = Input(3, 1)
          
          # Set outputs
          left_val = torch.tensor([[1.0], [2.0], [3.0]])
          right_val = torch.tensor([[4.0], [5.0], [6.0]])
          left_layer.set(left_val)
          right_layer.set(right_val)
          
          # sum layer
          sum_layer = Sum(left_layer, right_layer, rows=3, cols=1)

          
          # incoming grad
          grad_val = torch.tensor([[0.0], [1.0], [2.0]])
          sum_layer.grad = grad_val
          
          left_layer.clear_grad()
          right_layer.clear_grad()
          
          sum_layer.forward()
          
          # backwards pass
          sum_layer.backward()
          
          # true grad should be the passed in gradient
          expected_grad = torch.tensor([[0.0], [1.0], [2.0]])

          print("\nBackward Pass:")
          print("Left layer gradient:\n", left_layer.grad)
          print("Right layer gradient:\n", right_layer.grad)
          print("Expected gradient:\n", expected_grad)
          
          #actual tests
          print("\n\n Testing backwards Sum: ")
          self.assertTrue(torch.allclose(left_layer.grad, expected_grad),
                              f"Left layer gradient incorrect: expected {expected_grad}, got {left_layer.grad}")
          self.assertTrue(torch.allclose(right_layer.grad, expected_grad),
                              f"Right layer gradient incorrect: expected {expected_grad}, got {right_layer.grad}")


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
