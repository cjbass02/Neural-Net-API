import unittest
import torch
import numpy as np
import network  
from layers import Layer, Input, Regularization

class TestRegularization(unittest.TestCase):
    def test_forward(self):
        # Create a weight layer
        weight = Input(2, 2)
        weight_tensor = torch.tensor([[1.0, 2.0],
                                    [3.0, 4.0]])
        weight.set(weight_tensor)
        
        lambda_reg = 0.01
        reg = Regularization(weight, rows=1, cols=1, lambda_reg=lambda_reg)
        reg.forward()
        
        # 1^2 + 2^2 + 3^2 + 4^2 = 30.
        expected_loss = lambda_reg * 30.0
        print("TestRegularization: output =", reg.output.item(), "expected =", expected_loss)
        self.assertAlmostEqual(reg.output.item(), expected_loss, places=5)


    def test_regularization_backward(self):
        # Create W layer
        W_layer = Input(2, 3)
        W_val = torch.tensor([[0.1, 0.3, 0.5], 
                              [-0.2, -0.4, -0.6]])
        W_layer.set(W_val)
        
        # Define lamvda
        lambda_reg = 0.001
        # Create the reg layer
        reg_layer = Regularization(W_layer, rows=1, cols=1, lambda_reg=lambda_reg)
        
        # forward pass
        reg_layer.forward()
        
        # incoming gradient 
        grad_val = torch.tensor([[1.0]])
        reg_layer.grad = grad_val
        
        W_layer.clear_grad()
        
        #backward pass
        reg_layer.backward()
        
        # Expected gradient for W should be: incoming grad * lambda_reg * W_val. (all element wise since lambda and g are scalers :) )
        #expected_djdw = grad_val.item() * lambda_reg * W_val
        expected_djdw = torch.tensor([[ 0.0001,  0.0003,  0.0005],
                                      [-0.0002, -0.0004, -0.0006]])

        print("W_layer.grad:", W_layer.grad)
        print("Expected gradient:", expected_djdw)
        
        # Actual Tests
        self.assertTrue(torch.allclose(W_layer.grad, expected_djdw),
                        f"Incorrect gradient for weight layer: expected {expected_djdw}, got {W_layer.grad}")

if __name__ == '__main__':
    unittest.main()