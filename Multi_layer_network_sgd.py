import mnist
import numpy as np

class Neural_Network():
    def __init__(self,layers):
        self.layers = layers
        self.nlayers = len(layers)
        self.output_neurons = layers[self.nlayers-1]
        self.nfeatures = layers[0]
        self.weights = []
        self.biases = []
        for layer in range(1,self.nlayers):
            n_neurons = self.nfeatures
            # np.random.seed(0)
            # self.weights.append(np.random.rand(n_neurons,layers[layer]))
            # self.biases.append(np.array([np.ones(layers[layer])]).T)

            std_dev = np.sqrt(1 / (n_neurons + layers[layer]))  # Xavier initialization
            self.weights.append(np.random.normal(size=(n_neurons, layers[layer]), scale=std_dev))
            self.biases.append(np.random.normal(size=(layers[layer], 1), scale=std_dev))
            self.nfeatures = layers[layer]

        # print(f" initial weights are {self.weights}")

        # print(f" initial bias are {self.biases} ")

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def sigmoid_derivative(self,z):
        return z*(1-z)

    def cross_entropy_error(self,PredictVector, TargetVector):
        Logits = np.log(PredictVector)
        Values = np.dot(TargetVector.T, Logits)

        return (np.sum(Values))

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def softmax_stable(self,z):
        """Computes softmax function.
        z: array of input values.
        Returns an array of outputs with the same shape as z."""
        # For numerical stability: make the maximum of z's to be 0.
        shiftz = z - np.max(z)
        exps = np.exp(shiftz)
        return exps / np.sum(exps)


    def cross_entropy(self, pred_out, true_out):
        return -(np.sum((np.log2(pred_out) * true_out), axis = 1))

    def cross_entropy_delta(self,pred_out,true_out):
        return (pred_out - true_out)


    def to_categorical(self, training_outputs ):
        categorical_output = np.zeros((training_outputs.shape[0],self.output_neurons))

        training_outputs = np.array(training_outputs)

        for index in range(len(training_outputs)):
            value = training_outputs[index][0]
            categorical_output[index].put(value,1)

        return categorical_output


    def foreward_propagate(self,training_inputs ):

        estimated_output = np.empty((training_inputs.shape[0],self.output_neurons))
        activation_outputs = []
        activation_outputs.append(training_inputs)
        for (weight_index,weight) in enumerate(self.weights):
            if weight_index != len(self.weights)-1:
                linear_sum = np.dot(training_inputs, weight).T + self.biases[weight_index]
                activation_output = self.sigmoid(linear_sum)
                activation_outputs.append(activation_output.T)
                training_inputs = activation_output.T
            else:
                linear_sum = np.dot(training_inputs, weight).T + self.biases[weight_index]
                activation_output = self.softmax_stable(linear_sum).astype(np.float64)
                activation_outputs.append(activation_output.T)
            estimated_output = activation_output.T

        return estimated_output,activation_outputs

    def backpropagate(self, error, activation_outputs):
        del_weights = {}
        del_bias = {}
        deltas = {}
        deltas[self.nlayers-1] = error
        for layer_index in range(self.nlayers-1,0,-1):
            deltas[layer_index-1] = np.dot(deltas[layer_index], self.weights[layer_index-1].T) * self.sigmoid_derivative(activation_outputs[layer_index-1])

        # print(f" deltas elments shapes  are {deltas[0].shape} and {deltas[1].shape} and {deltas[2].shape}")

        for layer_index in range(self.nlayers-1,0,-1):
            del_bias[layer_index] = deltas[layer_index]
            del_weights[layer_index] = np.dot(activation_outputs[layer_index-1].T, deltas[layer_index])

        # print(f"weights to adjusted are {del_weights}  and shapes are {del_weights[1].shape} and {del_weights[2].shape}")
        # print(f"biases to adjusted  are {del_bias} and shapes are {del_bias[1].shape} and {del_bias[2].shape}")

        return del_weights, del_bias


    def learn(self, training_inputs,training_outputs,learning_rate):

        categorical_tranining_outputs = self.to_categorical(training_outputs)
        estimated_output,activation_outputs = self.foreward_propagate(training_inputs)
        delta = self.cross_entropy_delta(estimated_output,categorical_tranining_outputs)
        total_error = np.sum(delta)
        del_weights, del_bias = self.backpropagate(delta,activation_outputs)
        # total_error = self.cross_entropy_error(estimated_output,categorical_tranining_outputs)
        # print(total_error)
        final_weights = {}
        final_bias = {}
        for layer in range(1,self.nlayers):
            final_weights[layer] = self.weights[layer-1] - learning_rate * del_weights[layer]
            final_bias[layer] = self.biases[layer-1] - learning_rate * (np.mean(del_bias[layer], axis=0)).reshape(self.biases[layer-1].shape[0],1)

        return final_weights, final_bias, total_error

    def train(self,training_inputs,training_outputs,epochs, learning_rate ):

        for epoch in range(epochs):
            final_weights,final_bias,total_error = self.learn(training_inputs,training_outputs,learning_rate)
            final_weights_layer_list = list(final_weights.keys())
            final_bias_layer_list = list(final_bias.keys())
            for layer_index in final_weights_layer_list:
                self.weights[layer_index-1] = final_weights[layer_index]

            for layer_index in final_bias_layer_list:
                self.biases[layer_index-1] = final_bias[layer_index]

            # print(f"updated weight at epoch {epoch+1} is {self.weights} and shape is {self.weights[0].shape} and {self.weights[1].shape}")

            # print(f"updated bias at epoch {epoch+1} is {self.biases} and shape is {self.biases[0].shape} and {self.biases[1].shape}")

            print(f"total error at epoch {epoch+1} is {total_error}")

if __name__ == '__main__':

    x_train, y_train = mnist.train_images(), mnist.train_labels()
    nfeatures = x_train.shape[1] * x_train.shape[2]
    x_train_flatten = x_train.reshape((x_train.shape[0], nfeatures)).astype(np.float64)
    y_train = y_train.reshape((y_train.shape[0],1))
    x_test, y_test = mnist.test_images(),mnist.test_labels()
    x_test_flatten = x_test.reshape((x_test.shape[0],nfeatures)).astype(np.float64)
    y_test = y_test.reshape((y_test.shape[0], 1))
    epochs = 40
    learning_rate = 0.05
    n1_hidden = 100
    n_output = 10
    layers = [nfeatures,n1_hidden,n_output]
    neural_nework1 = Neural_Network(layers)
    neural_nework1.train(x_train_flatten,y_train,epochs,learning_rate)
