import unittest
import torch
import numpy as np
from layers import Input, Linear, ReLU, Regularization, MSELoss, Sum
from network import Network

class TestCompleteNetwork(unittest.TestCase):
    def test_forward(self):
        # Lab 2 values
        X_val = torch.tensor([[0.105, 0.815]])
        W1_val = torch.tensor([[0.75, -0.55, 0.25],
                                [0.25, -0.75, -0.15]], requires_grad=True) # require grad implimented for later labs
        b1_val = torch.tensor([-0.1, 0.2, -0.3], requires_grad=True)
        W2_val = torch.tensor([[0.1, -0.4],
                                [-0.2, 0.5],
                                [0.3, -0.6]], requires_grad=True)
        b2_val = torch.tensor([0.2, 0.1], requires_grad=True)
        Y_true = torch.tensor([[0.815, 0.105]])
        lambda_l2 = 0.001

        # Get expected output from model with 1 hidden layer (code from lab 2)
        Z1_expected = X_val @ W1_val + b1_val
        A1_expected = torch.clamp(Z1_expected, min=0)
        Z2_expected = A1_expected @ W2_val + b2_val
        Y_pred_expected = Z2_expected
        mse_loss_expected = torch.mean((Y_pred_expected - Y_true) ** 2)
        reg_loss_expected = lambda_l2 * (torch.sum(W1_val ** 2) + torch.sum(W2_val ** 2))
        total_loss_expected = mse_loss_expected + reg_loss_expected

        # build the same network from this labs implimetnawtion
        net = Network()
        
        # Input layer for x data
        input_layer = Input(1, 2)
        net.set_input(input_layer)
        input_layer.set(X_val)
        
        # Make W1 and b1
        W1_layer = Input(2, 3) 
        b1_layer = Input(1, 3)
        W1_layer.set(W1_val)
        b1_layer.set(b1_val.unsqueeze(0)) # unsqueeze does the same thing as keepdim
        
        # Combine x, w1, b1 with linear layer
        hidden_linear = Linear(1, input_layer, W1_layer, b1_layer, cols=3)
        net.add(hidden_linear)
        
        # Relu layer
        hidden_relu = ReLU(1, hidden_linear, cols=3)
        net.add(hidden_relu)
        
        # Make W2, and b2
        W2_layer = Input(3, 2) 
        b2_layer = Input(1, 2) 
        W2_layer.set(W2_val)
        b2_layer.set(b2_val.unsqueeze(0)) # Again... keepdim

        #Linear comb of the hidden layer
        output_linear = Linear(1, hidden_relu, W2_layer, b2_layer, cols=2)
        net.add(output_linear)
        net.set_output(output_linear)
        
        # call forward on the network to get preds
        Y_pred = net.forward(X_val)

        # MSE loss layer
        target_layer = Input(1, 2)
        target_layer.set(Y_true)
        mse_layer = MSELoss(output_linear, target_layer, rows=1, cols=1)
        net.add(mse_layer)

        
        
        # Reg layer
        reg_layer1 = Regularization(W = W1_layer, rows=1, cols=1, lambda_reg=lambda_l2)
        net.add(reg_layer1)
        reg_layer2 = Regularization(W = W2_layer, rows=1, cols=1, lambda_reg=lambda_l2)
        net.add(reg_layer2)


        # sum reg layers
        reg_sum = Sum(l_layer = reg_layer1, r_layer = reg_layer2)
        net.add(reg_sum)
        

        #sum mse and reg
        J = Sum(l_layer = reg_sum, r_layer = mse_layer)
        net.add(J)


        total_loss = net.forward(X_val)



        print("\n=== Test Network===")
        print("X =", X_val)
        print("Z1 (expected) =", Z1_expected)
        print("A1 (expected, after ReLU) =", A1_expected)
        print("Y_pred (network output) =", Y_pred)
        print("Y_pred (expected) =", Y_pred_expected)
        print("Total Loss (computed) =", total_loss.item(), "expected =", total_loss_expected.item())
        
        self.assertTrue(torch.allclose(Y_pred, Y_pred_expected, atol=1e-4))
        self.assertAlmostEqual(total_loss.item(), total_loss_expected.item(), places=3)

        


if __name__ == '__main__':
    unittest.main()