import numpy as np
inputs = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
targets = [[0,0],[0,1],[1,0],[1,1]]


class Layer:
    def __init__(self, neuron_number, inputs_number, is_output_layer):
        self.inputs_number = inputs_number
        self.is_output_layer = is_output_layer
        self.weights = 0.1 * np.random.randn(neuron_number, inputs_number)
        self.learning_rate = 0.1
        self.inputs = None
        self.output = None
        self.absolute_errors = None

    def forward_propagation(self):
        self.net = np.dot(self.inputs, self.weights.T)
        self.output = 1 / (1 + np.exp(-self.net))

    def backward_propagation(self, target=None, next_layer_absolute_error=None, next_layer_weights= None):
        if self.is_output_layer:
            error = target - self.output
            absolute_error = self.output * (1 - self.output) * error
            self.absolute_errors = absolute_error
            return self.absolute_errors
        else:
            weights_transposed = next_layer_weights
            dot_product_result = np.dot(next_layer_absolute_error, weights_transposed)
            absolute_errors = dot_product_result * (self.output * (1 - self.output))
            self.absolute_errors = absolute_errors
            return self.absolute_errors

    def update_weights(self):

        delta_weights = self.learning_rate * np.outer(self.absolute_errors, self.inputs)
        old_weights = self.weights.copy()
        self.weights += delta_weights
        return old_weights

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, neuron_number, inputs_number, is_output_layer):
        layer = Layer(neuron_number, inputs_number, is_output_layer)
        self.layers.append(layer)

    def forward_propagation(self, inputs):
        self.layers[0].inputs = inputs
        self.layers[0].forward_propagation()

        for i in range(1, len(self.layers)):
            self.layers[i].inputs = self.layers[i - 1].output
            self.layers[i].forward_propagation()

        return self.layers[-1].output

    def backward_propagation(self, target):
        output_absolute_error = self.layers[-1].backward_propagation(target=target)
        print(f"Output Layer Absolute Errors: {output_absolute_error}")
        total_error = 0
        for error in output_absolute_error:
            total_error = total_error + (error*error)
        total_error = total_error * 0.5

        print("\n\n")
        print("Total Error for this sample:")
        print(total_error)
        print("\n\n")

        for i in range(len(self.layers) - 2, -1, -1):
            next_layer_absolute_error = self.layers[i + 1].absolute_errors
            next_layer_weights = self.layers[i + 1].weights
            layer_absolute_error = self.layers[i].backward_propagation(
                next_layer_absolute_error=next_layer_absolute_error,
                next_layer_weights=next_layer_weights)
            print(f"Hidden Layer {i + 1} Absolute Errors: {layer_absolute_error}")

        for i, layer in enumerate(self.layers):
            old_weights = layer.update_weights()
            print(f"Layer {i + 1} Old Weights:\n{old_weights}")
            print(f"Layer {i + 1} Updated Weights:\n{layer.weights}")

        return total_error

    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            mse_error = 0
            for input_data, target_data in zip(inputs, targets):
                output = self.forward_propagation(input_data)
                te = self.backward_propagation(target_data)
                mse_error = te + mse_error
                print(f"TOTAL TOTAL Error for Epoch {epoch} = TE1 + TE2 + TE3 + TE4 : {mse_error}")
                print(f"Also MSE for Epoch {epoch} = TOTAL TOTAL ERROR / SAMPLE COUNT : {mse_error * 0.25}")



network = NeuralNetwork()

decision = 0
i = 1

while decision == 0:
    neuron_number = int(input(f"How many neurons dost thou wisheth in the {i}th layer?"))
    input_number = int(input(f"How many inputs art there for the {i}th layer?"))
    is_out = input(f"Is it the output stratum?")

    if is_out == "True":
        network.add_layer(neuron_number=neuron_number, inputs_number=input_number, is_output_layer=True)
        epoch_number = int(input(f"How many neurons dost thou wisheth for training"))
        decision = 1
    else:
        network.add_layer(neuron_number=neuron_number, inputs_number=input_number, is_output_layer=False)
    i = i+1

network.train(inputs, targets, epochs=epoch_number)