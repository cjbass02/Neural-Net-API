import unittest
import torch
import numpy as np
import network  
from layers import Layer, Input, Softmax

class TestSoftmax(unittest.TestCase):
    def test_forward(self):
        # Create logits
        x_vals = torch.tensor([[1, 2, 3],
                                [1, 2, 3]], dtype=torch.float32)
        # Create labels tensor 
        y_vals = torch.tensor([[0, 0, 1],
                                [1, 0, 0]], dtype=torch.float32)

        # Create the actual layers
        x_layer = Input(2, 3)
        x_layer.set(x_vals)
        
        y_layer = Input(2, 3)
        y_layer.set(y_vals)
        
        # softmax layer
        softmax = Softmax(x_layer, y_layer, rows=1, cols=1)
        loss = softmax.forward()

        # Compute expected probs and loss
        shifted_logits = x_vals - torch.max(x_vals, dim=1, keepdim=True)[0]
        exp_scores = torch.exp(shifted_logits)
        probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)
        epsilon = 1e-8
        correct_logprobs = -torch.log(torch.sum(probs * y_vals, dim=1) + epsilon)
        expected_loss = torch.mean(correct_logprobs)
        
        print("TestSoftmax: loss =", loss.item(), "expected =", expected_loss.item())
        print("TestSoftmax: classifications =", softmax.classifications, "expected =", probs)
        
        # tests
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)
        self.assertTrue(torch.allclose(softmax.classifications, probs, atol=1e-5))


    def test_softmax_backward(self):
        # Create input layers for logits and preds
        logits_layer = Input(1, 3)
        labels_layer = Input(1, 3)
        
        # prob vlaues
        # Logits for one sample with three classes
        logits_val = torch.tensor([[2.0, 1.0, 0.1]])
        # Target labels (one-hot encoded): class 0 is the correct class
        labels_val = torch.tensor([[1.0, 0.0, 0.0]])
        
        logits_layer.set(logits_val)
        labels_layer.set(labels_val)
        
        # Create the softmax layer
        softmax_layer = Softmax(logits_layer, labels_layer, rows=1, cols=1)
        
        # Forward pass
        loss = softmax_layer.forward()
        
        # an incoming gradient
        softmax_layer.grad = torch.tensor(1.0)
        
        logits_layer.clear_grad()
        
        # backwards pass
        softmax_layer.backward()
        
        # expected gradient for logits (g = 1, so there is no need to add it)
        #expected_grad = softmax_layer.classifications - labels_layer.output
        expected_grad = torch.tensor([-0.341, 0.242, 0.099])
        
        print("\nBackward Pass:")
        print("Computed gradient for logits (x):\n", logits_layer.grad)
        print("Expected gradient for logits (x):\n", expected_grad)
        
        # tests
        self.assertTrue(torch.allclose(logits_layer.grad, expected_grad, rtol=0.1),
                        f"Incorrect gradient for logits: expected {expected_grad}, got {logits_layer.grad}")



if __name__ == '__main__':
    unittest.main()