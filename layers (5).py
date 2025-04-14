import torch
from numpy import newaxis as np_newaxis
import numpy as np

# TODO: Please be sure to read the comments in the main lab and think about your design before
# you begin to implement this part of the lab.

# Layers in this file are arranged in roughly the order they
# would appear in a network.

# MAKE CHANGES TO OUTPUT SIZE ERRORS. DO IT IN EACH ASSIGNMENT OF OUTPUT

class Layer:
    def __init__(self, rows, cols):
        """
        TODO: Add arguments and initialize instance attributes here.
        """
        self.num_nodes = rows*cols
        self.rows = rows
        self.cols = cols
        self.output = torch.zeros(rows, cols)
        self.grad = torch.zeros(self.rows, self.cols)

    def accumulate_grad(self, grad):
        """
        TODO: Add arguments as needed for this method.
        This method should accumulate its grad attribute with the value provided.
        """
        self.grad += grad

    def clear_grad(self):
        """
        TODO: Add arguments as needed for this method.
        This method should clear grad elements. It should set the grad to the right shape 
        filled with zeros.
        """
        self.grad = torch.zeros(self.rows, self.cols)

    def step(self, learning_rate):
        """
        TODO: Add arguments as needed for this method.
        Most tensors do nothing during a step so we simply do nothing in the default case.
        """
        pass

class Input(Layer):
    def __init__(self, rows, cols):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, rows, cols) # TODO: Pass along any arguments to the parent's initializer here.

    def set(self, output):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set the output of this input layer.
        :param output: The output to set, as a torch tensor. Raise an error if this output's size
                       would change.
        """
        expected_shape = (self.rows, self.cols)
        if output.shape != expected_shape:
            raise ValueError(f"Expected output shape {expected_shape}, but got {output.shape}.")
        self.output = output

    def randomize(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set the output of this input layer to random values sampled from the standard normal
        distribution (torch has a nice method to do this). Ensure that the output does not
        change size.
        """
        random_val = torch.randn(self.rows, self.cols) * 0.1

        self.output = random_val

    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        #does nothing!
        pass

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        This method does nothing as the Input layer should have already received its output
        gradient from the previous layer(s) before this method was called.
        """
        # Does nothing!
        pass

    def step(self, learning_rate):
        """
        TODO: Add arguments as needed for this method.
        This method should have a precondition that the gradients have already been computed
        for a given batch.

        It should perform one step of stochastic gradient descent, updating the weights of
        this layer's output based on the gradients that were computed and a learning rate.
        """
        self.output = self.output - (learning_rate*self.grad)


class Linear(Layer):
    def __init__(self, rows, X, W, b, cols = 1):
        """
        TODO: Accept any arguments specific to this child class.
        """
        # prev layers should be in order of X, W, b
        Layer.__init__(self, rows, cols) # TODO: Pass along any arguments to the parent's initializer here.
        self.X = X
        self.W = W
        self.b = b

    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """

        Z = self.X.output @ self.W.output + self.b.output
        expected_shape = (self.rows, self.cols)
        # if Z.shape != expected_shape:
        #     raise ValueError(f"Expected output shape {expected_shape}, but got {output.shape}.")
        self.output = Z


    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        # self.grad should contain dj/du
        # calculate dj/dx = W^T g
        djdx = self.grad @ self.W.output.T
        #djdx = self.W.output.T @ self.grad 

        # calculate dj/dw = g X^T
        djdw = self.X.output.T @ self.grad
        #djdw = self.grad @ self.X.output.T

        # calculate dj/db
        #djdb = self.grad
        djdb = torch.sum(self.grad, dim=0, keepdim=True)

        #send em off
        self.W.accumulate_grad(djdw)
        self.X.accumulate_grad(djdx)
        self.b.accumulate_grad(djdb)



class ReLU(Layer):
    # There should only ever be 1 column in a relu layer
    def __init__(self, rows, prev_output, cols = 1):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, rows, cols) # TODO: Pass along any arguments to the parent's initializer here.
        self.prev_output = prev_output

    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        input_tensor = self.prev_output.output
        # expected_shape = (self.rows, self.cols)
        # if torch.clamp(input_tensor, min=0).shape != expected_shape:
        #     raise ValueError(f"Expected output shape {expected_shape}, but got {output.shape}.")
        self.output = torch.clamp(input_tensor, min=0)

    def relu_prime(self, x):
        """
        Simulates the derivative of the relu function to help with backprop
        """
        return (x > 0).float()
        

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        # calculate du/dy, where y is self.previous output
        djdy = self.grad * self.relu_prime(self.prev_output.output)

        # send the new gradient off
        self.prev_output.accumulate_grad(djdy)


class MSELoss(Layer):
    """
    This is a good loss function for regression problems.

    It implements the MSE norm of the inputs.
    """
    # Output should always be a scalar
    def __init__(self, pred, targets, rows = 1, cols = 1):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, rows, cols) # TODO: Pass along any arguments to the parent's initializer here.
        self.pred = pred
        self.targets = targets

    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        # Compute the error and then its squared mean.
        error = self.pred.output - self.targets.output
        mse = torch.mean(error ** 2)

        # Set the layer's output to the computed MSE.
        # expected_shape = (self.rows, self.cols)
        # if mse.shape != expected_shape:
        #     raise ValueError(f"Expected output shape {expected_shape}, but got {output.shape}.")
        self.output = mse


    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        #calculate dj/dy_pred = g (.) (y_pred - y_true)
        djdy_pred = self.grad * (self.pred.output - self.targets.output)

        #send the gradient off to the layer where the preds came out
        self.pred.accumulate_grad(djdy_pred)


class Regularization(Layer):
    def __init__(self, W, rows = 1, cols = 1, lambda_reg = 0.01):
        """
        TODO: Accept any arguments specific to this child class.
        Parameters: Ws is a list of all weight layers
        """
        Layer.__init__(self, rows, cols) # TODO: Pass along any arguments to the parent's initializer here.
        self.lambda_reg = lambda_reg
        self.W = W

    def helper_frobinius(self, W):
        """
        Caclulates the frobinius norm of a matrix
        """
        return torch.norm(W, p='fro')

    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        reg_loss = 0.0

        frob_norm = self.helper_frobinius(self.W.output)
        reg_loss += frob_norm**2

        expected_shape = (self.rows, self.cols)
        self.output = self.lambda_reg * (reg_loss)


    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        # calculate dj/dw = g*lambda*w
        djdw = self.grad * self.lambda_reg * self.W.output

        #send the gradient off
        self.W.accumulate_grad(djdw)


class Softmax(Layer):
    """
    This layer is an unusual layer.  It combines the Softmax activation and the cross-
    entropy loss into a single layer.

    The reason we do this is because of how the backpropagation equations are derived.
    It is actually rather challenging to separate the derivatives of the softmax from
    the derivatives of the cross-entropy loss.

    So this layer simply computes the derivatives for both the softmax and the cross-entropy
    at the same time.

    But at the same time, it has two outputs: The loss, used for backpropagation, and
    the classifications, used at runtime when training the network.

    TODO: Create a self.classifications property that contains the classification output,
    and use self.output for the loss output.

    See https://www.d2l.ai/chapter_linear-networks/softmax-regression.html#loss-function
    in our textbook.

    Another unusual thing about this layer is that it does NOT compute the gradients in y.
    We don't need these gradients for this lab, and usually care about them in real applications,
    but it is an inconsistency from the rest of the lab.
    """
    def __init__(self,  x , y, rows = 1, cols = 1):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, rows, cols) # TODO: Pass along any arguments to the parent's initializer here.
        self.classifications = None
        self.x = x
        self.y = y

    def forward(self):
        """
        Computes the softmax probabilities and the cross-entropy loss.
        """
        
        shifted_logits = self.x.output - torch.max(self.x.output, dim=1, keepdim=True)[0]
        exp_scores = torch.exp(shifted_logits)
        probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)
        self.classifications = probs

        # Compute cross-entropy loss
        epsilon = 1e-8
        #correct_logprobs = -torch.log(torch.sum(probs * self.y.output, dim=1) + epsilon)
        correct_logprobs = -torch.sum(self.y.output * torch.log(probs), dim = 1)
        loss = torch.mean(correct_logprobs)
        self.output = loss

        return self.output

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        #calculate dj/dz = g(p-t)
        djdz = self.grad * (self.classifications - self.y.output)
        self.x.accumulate_grad(djdz)


class Sum(Layer):
    def __init__(self, l_layer, r_layer, rows = 1, cols = 1):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, rows, cols) # TODO: Pass along any arguments to the parent's initializer here.
        self.l_layer = l_layer
        self.r_layer = r_layer

    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        self.output = self.r_layer.output + self.l_layer.output

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        djdx = self.grad
        self.l_layer.accumulate_grad(djdx)
        self.r_layer.accumulate_grad(djdx)


