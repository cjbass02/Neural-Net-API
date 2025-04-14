import unittest
import torch
import numpy as np
import network  
from layers import Layer, Input, Linear, ReLU, MSELoss, Regularization, Softmax, Sum

class TestLayer(unittest.TestCase):
    def test_layer_init(self):
        rows, cols = 3, 4
        layer = Layer(rows, cols)
        expected_output = torch.zeros(rows, cols)
        print("TestLayer: output =", layer.output, "expected =", expected_output)
        self.assertEqual(layer.num_nodes, rows * cols)
        self.assertEqual(layer.rows, rows)
        self.assertEqual(layer.cols, cols)
        self.assertTrue(torch.equal(layer.output, expected_output))


if __name__ == '__main__':
    unittest.main()
