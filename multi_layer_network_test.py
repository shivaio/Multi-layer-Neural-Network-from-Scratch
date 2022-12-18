import math

import mnist
import numpy as np

class Neural_Network():
    def __init__(self,layers, hidden_activation, output_activation, loss_function, optimizers):
        self.layers = layers
        self.nlayers = len(layers)
        self.output_neurons = layers[self.nlayers-1]
        self.nfeatures = layers[0]
        self.weights = []
        self.biases = []
        for layer in range(1,self.nlayers):
            n_neurons = self.nfeatures
            np.random.seed(0)
            self.weights.append(np.random.rand(n_neurons,layers[layer]))
            self.biases.append(np.array([np.ones(layers[layer])]).T)
            self.nfeatures = layers[layer]
        # print("--------------")
        # print(weights)
        # print("--------------")
        # print(biases)
        # print("--------------")
        for weight in self.weights:
            print("--------------")
            print(weight.shape)
            print("--------------")
        for bias in self.biases:
            print("--------------")
            print(bias.shape)
            print("--------------")


    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def sigmoid_derivative(self,z):
        return z*(1-z)

    def softmax_derivative(self,z):
            s = self.softmax(z)
            D = -np.outer(s, s) + np.diag(s.flatten())
            return D

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    # def softmax_derivative(self,z):
    #     s = self.softmax(z)
    #     D = np.stack([np.diag(s[i, :]) for i in range(s.shape[0])], axis=0)
    #     comb = np.matmul(np.expand_dims(s, 2), np.expand_dims(s, 1))
    #     return D - comb

    # def softmax(self,z):
    #     return np.exp(z)/np.sum(np.exp(z), axis = 0)

    # def softmax(x):
    #     EPS = np.finfo(np.float64).eps
    #     x = x - x.max(axis=1).reshape((-1, 1))
    #     exp = np.exp(x)
    #     s = np.sum(exp, axis=1).reshape((-1, 1))
    #     return exp / (s + EPS)

    def cross_entropy(self, pred_out, true_out):
        return -(np.sum((np.log2(pred_out) * true_out), axis = 1))

    def cross_entropy_delta(self,pred_out,true_out):
        return (true_out - pred_out)


    # def d_categorical_crossentropy(self,y_pred, y_true):
    #     return np.expand_dims(-y_true / y_pred , 1)

    def cross_entropy_loss(self,p, y):
        """Cross-entropy loss between predicted and expected probabilities.
        p: vector of predicted probabilities.
        y: vector of expected probabilities. Has to be the same shape as p.
        Returns a scalar.
        """
        assert (p.shape == y.shape)
        return -np.sum(y * np.log(p))

    def new_cross_entropy(self,X, y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector.
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        p = self.softmax(X)
        # We use multidimensional array indexing to extract
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def delta_cross_entropy(self,X, y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector.
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        grad = self.softmax(X)
        grad[range(m), y] -= 1
        grad = grad / m
        return grad

    def to_categorical(self, training_outputs ):
        # class_list = list(range(self.output_neurons))
        categorical_output = np.zeros((training_outputs.shape[0],self.output_neurons))

        # print(type(training_outputs))
        training_outputs = np.array(training_outputs)
        # print(training_outputs[0][0])
        for index in range(len(training_outputs)):
            value = training_outputs[index][0]
            # print(f"value is {value}")
            # print(categorical_output[index])
            categorical_output[index].put(value,1)
            # print("$$$$$$$$$$$$$$$$$")
            # print(categorical_output[index][value])
            # print("$$$$$$$$$$$$$$$$$")
            # break

        print("$$$$$$$$$$$$$$$$$")
        print(categorical_output)
        print(categorical_output.shape)
        print("$$$$$$$$$$$$$$$$$")

        return categorical_output


    def foreward_propagate(self,training_inputs ):

        estimated_output = np.empty((training_inputs.shape[0],self.output_neurons))
        activation_outputs = []
        activation_outputs.append(training_inputs)
        for (weight_index,weight) in enumerate(self.weights):
            if weight_index != len(self.weights)-1:
                activation_output = training_inputs
                # linear_sum = np.dot(training_inputs,weight).T + self.biases[weight_index]
                linear_sum = np.dot(training_inputs, weight).T
                print(f"linear sum with out bias is {linear_sum}")
                linear_sum = np.dot(training_inputs, weight).T + self.biases[weight_index]
                print(f"linear sum with bias is {linear_sum}")
                activation_output = self.sigmoid(linear_sum)
                activation_outputs.append(activation_output.T)
                # print(linear_sum)
                print(linear_sum.shape)
                print("====================================")
                print(activation_output)
                print("====================================")
                print(activation_output.shape)
                training_inputs = activation_output.T
            else:
                linear_sum = np.dot(training_inputs, weight).T + self.biases[weight_index]
                activation_output = self.softmax(linear_sum).astype(np.float64)
                activation_outputs.append(activation_output.T)
            estimated_output = activation_output.T
        print("---------------------------------------------")
        print(estimated_output)
        print(estimated_output.shape)
        print("---------------------------------------------")
        print(f"activation outputs are {activation_outputs}  and {len(activation_outputs)} and first layer shape is {activation_outputs[0].shape} and second layer shape is {activation_outputs[1].shape} ")
        return estimated_output,activation_outputs

    def backpropagate(self, error, activation_outputs):
        del_weights = {}
        del_bias = {}
        deltas = {}
        deltas[self.nlayers-1] = error
        for layer_index in range(self.nlayers-1,0,-1):
            # print(f"softmax derivative of activation {self.sigmoid_derivative(activation_outputs[layer_index - 1])}")
            # print(f"shape of derivative is {self.sigmoid_derivative(activation_outputs[layer_index - 1]).shape}")
            deltas[layer_index-1] = np.dot(deltas[layer_index], self.weights[layer_index-1].T) * self.sigmoid_derivative(activation_outputs[layer_index-1])
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print(f"sigmoid derivative of activation {self.sigmoid_derivative(activation_outputs[layer_index-1])}")
            print(f"delta is {deltas}")
            # print(f" shape are {deltas[0].shape},and {deltas[1].shape} and {deltas[2].shape}")
            print(f" keys of dict are {deltas.keys()}")
            # print(f"delta.shape is {deltas.shape}")
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        print(f" shape are {deltas[0].shape},and {deltas[1].shape} and {deltas[2].shape}")

        for layer_index in range(self.nlayers-1,0,-1):
            del_bias[layer_index] = deltas[layer_index]
            del_weights[layer_index] = np.dot(activation_outputs[layer_index-1].T, deltas[layer_index])

        print(f"weights to updated are {del_weights}  and shapes are {del_weights[1].shape} and {del_weights[2].shape}")
        print(f"biases to updated  are {del_bias} and shapes are {del_bias[1].shape} and {del_bias[2].shape}")

        return del_weights, del_bias




            # if layer_index == self.nlayers - 1:
            #     print("check check")
            #     print(activation_outputs[layer_index-1][0])
            #     print(activation_outputs[layer_index - 1][0].shape)
            #     print(self.softmax_derivative(activation_outputs[layer_index-1][0]))
            #     print(self.softmax_derivative(activation_outputs[layer_index-1][0]).shape)
            #     # error = np.dot(error.T, self.weights[layer_index-1])
            #     # print(error)
            #     # print(error.shape)
            #     # print("check check")
            #     # error = np.dot(error.T,activation_outputs[layer_index-2])
            #     # print(f"final error is {error}")
            #     # print(f"error.shape is {error.shape}")
            #     break
            # if layer_index == self.nlayers - 1:
            #     error = np.matmul(error, self.softmax_derivative(activation_outputs[layer_index-1][0]))
            #     print(f"error is {error} and shape is {error.shape}")
            #     error = np.squeeze(error, 1)
            #     print(f"error is {error} and shape is {error.shape}")
            #
            # else:
            #     error = error * self.sigmoid_derivative(activation_outputs[layer_index-2][0])
            # d_w = np.matmul(activation_outputs[layer_index - 1][1].transpose(), error)
            # d_b = np.mean(error, axis=0)
            # del_weights.append(d_w)
            # del_bias.append(d_b)
            # error = np.matmul(error, self.weights[layer_index - 1].transpose())



        # del_weights.reverse()
        # del_bias.reverse()
        # print(f"del weights are {del_weights}  ")
        # print(f"del bias are {del_bias}")





    def learn(self, training_inputs,training_outputs,learning_rate):

        categorical_tranining_outputs = self.to_categorical(training_outputs)
        print(f"categorical training outputs shape is {categorical_tranining_outputs.shape}")
        estimated_output,activation_outputs = self.foreward_propagate(training_inputs)
        print(f"estimated output shape is {estimated_output.shape}")
        error = self.cross_entropy_delta(estimated_output,categorical_tranining_outputs)
        # error = self.delta_cross_entropy(estimated_output, training_outputs)
        # error = np.reshape(error,(training_outputs.shape[0],1))
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # print(f"error is {error}")
        # print(error.shape)
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # error = self.delta_cross_entropy(estimated_output,training_outputs)
        # print(error)
        # print(error.shape)
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        # error = self.d_categorical_crossentropy(estimated_output,categorical_tranining_outputs)
        print("check")
        print(f"final error is {error} and shape {error.shape}")
        del_weights, del_bias = self.backpropagate(error,activation_outputs)
        final_weights = {}
        final_bias = {}
        for layer in range(1,self.nlayers):
            final_weights[layer] = self.weights[layer-1] - learning_rate * del_weights[layer]
            # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            # print(np.mean(del_bias[layer], axis=0).shape)
            # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            final_bias[layer] = self.biases[layer-1] - learning_rate * (np.mean(del_bias[layer], axis=0)).reshape(self.biases[layer-1].shape[0],1)

        print(f"final weights are {final_weights} and keys are {final_weights.keys()} and values are {final_weights.values()} and their shapes are {final_weights[1].shape} and {final_weights[2].shape}")
        print(f" final bias are {final_bias} and shapes are {final_bias[1].shape} and {final_bias[2].shape}")
        # return final_weights

    def train(self,training_inputs,training_outputs,epochs, learning_rate ):
        # for epoch in range(epochs):
            # final_weights = \
        self.learn(training_inputs,training_outputs,learning_rate)


if __name__ == '__main__':

    x_train, y_train = mnist.train_images(), mnist.train_labels()
    nfeatures = x_train.shape[1] * x_train.shape[2]
    # print(nfeatures)
    x_train_flatten = x_train.reshape((x_train.shape[0], nfeatures)).astype(np.float64)
    print(x_train_flatten.shape)
    # print(list(set(y_train)))
    y_train = y_train.reshape((y_train.shape[0],1))
    print(y_train.shape)
    x_test, y_test = mnist.test_images(),mnist.test_labels()
    x_test_flatten = x_test.reshape((x_test.shape[0],nfeatures)).astype(np.float64)
    # print(x_test_flatten[0])



    epochs = 10
    learning_rate = 0.05
    n1_hidden = 100
    # n2_hidden = 200
    # n3_hidden = 100
    n_output = 10
    layers = [nfeatures,n1_hidden,n_output]
    hidden_activation = []
    output_activation = []
    loss_function = []
    optimizer = []
    neural_nework1 = Neural_Network(layers,hidden_activation, output_activation, loss_function, optimizer)
    neural_nework1.train(x_train_flatten,y_train,epochs,learning_rate)
