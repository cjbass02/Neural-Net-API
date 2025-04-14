import unittest
import torch
import numpy as np
import network
from layers import Layer, Input

class TestStep(unittest.TestCase):
    def test_step(self):
        # Create a weights layer
        weight_layer = Input(3, 2)
        initial_weights = torch.tensor([[0.3, -0.5],
                                        [0.7, -0.9],
                                        [1.0, -0.1]])
        weight_layer.set(initial_weights.clone())
        
        # create gradient for the weight layer
        grad_weights = torch.tensor([[0.05, -0.1],
                                    [0.03, 0.01],
                                    [0.1, -0.2]])
        weight_layer.grad = grad_weights
        
        # learning rate
        learning_rate = 0.1
        
        # Call step
        weight_layer.step(learning_rate)
        
        # expected weights
        #expected_weights = initial_weights - (learning_rate * grad_weights)
        expected_weights = torch. tensor([[ 0.2950, -0.4900],
                                            [ 0.6970, -0.9010],
                                            [ 0.9900, -0.0800]])

        print("Expected new weights: " + str(expected_weights))
        print("Actual calculated new weights: " + str(weight_layer.output))
        
        self.assertTrue(torch.allclose(weight_layer.output, expected_weights, atol=1e-5),
                        f"Step update incorrect: expected {expected_weights}, got {weight_layer.output}")
    
if __name__ == '__main__':
    unittest.main()