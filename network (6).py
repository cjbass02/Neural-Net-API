import torch 
class Network:
    def __init__(self):
        """
        TODO: Initialize a `layers` attribute to hold all the layers in the gradient tape.
        """
        self.layers = []
        self.input_layer = None
        self.output_layer = None

    def add(self, layer):
        """
        Adds a new layer to the network.

        Sublayers can *only* be added after their inputs have been added.
        (In other words, the DAG of the graph must be flattened and added in order from input to output)
        :param layer: The sublayer to be added
        """
        # TODO: Implement this method.
        self.layers.append(layer)

    def set_input(self,input_data):
        """
        :param input: The sublayer that represents the signal input (e.g., the image to be classified)
        """
        # TODO: Delete or implement this method. (Implementing this method is optional, but do not
        # leave it as a stub.)
        self.input_layer = input_data

    def set_output(self,output):
        """
        :param output: SubLayer that produces the useful output (e.g., clasification decisions) as its output.
        """
        # TODO: Delete or implement this method. (Implementing this method is optional, but do not
        # leave it as a stub.)
        #
        # This becomes messier when your output is the variable o from the middle of the Softmax
        # layer -- I used try/catch on accessing the layer.classifications variable.
        # when trying to access read the output layer's variable -- and that ended up being in a
        # different method than this one.
        self.output_layer = output

    def forward(self,input_data):
        """
        Compute the output of the network in the forward direction, working through the gradient
        tape forward

        :param input: A torch tensor that will serve as the input for this forward pass
        :return: A torch tensor with useful output (e.g., the softmax decisions)
        """
        # TODO: Implement this method
        # TODO: Either remove the input option and output options, or if you keep them, assign the
        #  input to the input layer's output before performing the forward evaluation of the network.
        #
        # Users will be expected to add layers to the network in the order they are evaluated, so
        # this method can simply call the forward method for each layer in order.

        self.input_layer.output = input_data
        # Each layers forward method has no inputs, so we can just do this:
        for layer in self.layers:
            layer.forward()

        return self.layers[-1].output

    def backward(self):
        """
        Compute the gradient of the output of all layers through backpropagation backward through the 
        gradient tape.
        """
        # Initialize final gradient (J = 1)
        self.layers[-1].grad = torch.tensor(1.0)
        # go through gradients in reverse order
        for layer in reversed(self.layers):
            layer.backward()

    def step(self, learning_rate):
        """
        Perform one step of the stochastic gradient descent algorithm
        based on the gradients that were previously computed by backward, updating all learnable parameters 
        """
        # Pretty simple. Since most of the layers dont step, the basic parent class just does nothing for most layers
        for layer in self.layers:
            layer.step(learning_rate)
        
        # reset gradients
        for layer in self.layers:
            layer.clear_grad()
