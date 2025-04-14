import unittest
import torch
import numpy as np
import network  
from layers import Layer, Input, Linear


class TestLinear(unittest.TestCase):
    def test_forward(self):
        # Create layers for input, weight, and bias.
        X_layer = Input(2, 2)
        W_layer = Input(2, 2)
        b_layer = Input(2, 2)
        # Set values
        X_val = torch.tensor([[1.0, 2.0],
                              [3.0, 4.0]])
        W_val = torch.tensor([[1.0, 0.0],
                              [0.0, 1.0]])
        b_val = torch.tensor([[1.0, 1.0],
                              [1.0, 1.0]])
        X_layer.set(X_val)
        W_layer.set(W_val)
        b_layer.set(b_val)
        linear = Linear(2, X_layer, W_layer, b_layer, cols=2)
        linear.forward()
        expected = X_val @ W_val + b_val
        print("\n\nTestLinear Forward: output =", linear.output, "expected =", expected)
        self.assertTrue(torch.allclose(linear.output, expected))




    def test_backward(self):
        # Create layers for x w and b
        X_layer = Input(1, 2)
        W_layer = Input(2, 3)
        b_layer = Input(1, 3)

        # Set values
        X_val = torch.tensor([[1.0, 1.5]])
        W_val = torch.tensor([[0.1, 0.3, 0.5], 
                              [-0.2, -0.4, -0.6]])
        b_val = torch.tensor([[-0.2, 0.1, 0.2]])
        X_layer.set(X_val)
        W_layer.set(W_val)
        b_layer.set(b_val)

        # Create the linear layer
        linear = Linear(2, X_layer, W_layer, b_layer, cols=2)
        
        # Forward pass
        linear.forward()
        
        # incoming gradient 
        grad_val = torch.tensor([[1.0, 0.0, 1.0]])
        linear.grad = grad_val

        X_layer.clear_grad()
        W_layer.clear_grad()
        b_layer.clear_grad()

        # call backwards on the linear layer
        linear.backward()
        
        # expected grads
        #expected_djdx = grad_val @ W_val.T
        expected_djdx = torch.tensor([[0.6, -0.8]])
        # dj/dw 
        # expected_djdw =  X_val.T @ grad_val
        expected_djdw =  torch.tensor([[1.0, 1.5], [0.0, 0.0], [1.0, 1.5]]).T
        # dj/db = grad_val
        #expected_djdb = grad_val
        expected_djdb = torch.tensor([[1.0, 0.0, 1.0]])

        print("Test Linear backward:")
        print("X_layer grad:", X_layer.grad, "Expected:", expected_djdx)
        print("W_layer grad:", W_layer.grad, "Expected:", expected_djdw)
        print("b_layer grad:", b_layer.grad, "Expected:", expected_djdb)

        # Actual Tests
        self.assertTrue(torch.allclose(X_layer.grad, expected_djdx),
                        f"X_layer grad incorrect: expected {expected_djdx}, got {X_layer.grad}")
        self.assertTrue(torch.allclose(W_layer.grad, expected_djdw),
                        f"W_layer grad incorrect: expected {expected_djdw}, got {W_layer.grad}")
        self.assertTrue(torch.allclose(b_layer.grad, expected_djdb),
                        f"b_layer grad incorrect: expected {expected_djdb}, got {b_layer.grad}")

if __name__ == '__main__':
    unittest.main()