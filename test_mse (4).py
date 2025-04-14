import unittest
import torch
import numpy as np
import network  
from layers import Layer, Input, MSELoss

class TestMSELoss(unittest.TestCase):
    def test_forward(self):
        # Create  prediction and true layers
        pred = Input(2, 1)
        targets = Input(2, 1)
        pred_tensor = torch.tensor([[2.0],
                                    [4.0]])
        target_tensor = torch.tensor([[1.0],
                                      [3.0]])
        pred.set(pred_tensor)
        targets.set(target_tensor)
 
        mse_loss = MSELoss(pred, targets, rows=2, cols=1)
        mse_loss.forward()
        error = pred_tensor - target_tensor
        expected_mse = torch.mean(error ** 2)
        print("TestMSELoss: output =", mse_loss.output.item(), "expected =", expected_mse.item())
        self.assertAlmostEqual(mse_loss.output.item(), expected_mse.item(), places=5)


    def test_mse_loss_backward(self):
        # Input layers (target and preds) for MSE
        pred_layer = Input(4, 1)
        target_layer = Input(4, 1)
        
        # Set outputs
        pred_val = torch.tensor([[-1.0], [0.0], [2.0], [3.0]])
        target_val = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
        pred_layer.set(pred_val)
        target_layer.set(target_val)
        
        # Create MSE layer
        mse_loss = MSELoss(pred_layer, target_layer, rows=1, cols=1)
        
        # Forward pass
        mse_loss.forward()
        
        # incoming grad
        grad_val = torch.tensor([[1.0]])
        mse_loss.grad = grad_val
        pred_layer.clear_grad()
        
        # backward pass
        mse_loss.backward()
        
        # expected gradient for the preds
        #expected_grad = grad_val * (pred_val - target_val)
        expected_grad = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
        

        print("\nBackward Pass:")
        print("Incoming gradient (mse_loss.grad):", mse_loss.grad)
        print("Computed gradient for predictions:\n", pred_layer.grad)
        print("Expected gradient for predictions:\n", expected_grad)
        
        # Actual test assertions.
        print("\n\n Testing backwards MSELoss: ")
        self.assertTrue(torch.allclose(pred_layer.grad, expected_grad),
                        f"Incorrect gradient for predictions: expected {expected_grad}, got {pred_layer.grad}")


if __name__ == '__main__':
    unittest.main()