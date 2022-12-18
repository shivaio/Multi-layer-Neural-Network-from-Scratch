import mnist
import numpy as np
import matplotlib.pyplot as plt

class Neural_Network():

    def __init__(self,layers):
        self.layers = layers
        self.nlayers = len(layers)
        self.output_neurons = layers[self.nlayers-1]
        self.nfeatures = layers[0]
        self.weights = []
        self.biases = []
        self.check_weights = []

        for layer in range(1,self.nlayers):
            n_neurons = self.nfeatures
            std_dev = np.sqrt(1 / (n_neurons + layers[layer]))
            self.weights.append(np.random.normal(size=(n_neurons, layers[layer]), scale=std_dev))
            self.biases.append(np.random.normal(size=(layers[layer],1), scale=std_dev))
            self.check_weights.append(np.random.normal(size=(n_neurons, layers[layer]), scale=std_dev))
            self.nfeatures = layers[layer]


    def sigmoid(self,z):

       if np.all(z) < 0:
           return np.exp(z)/(1+np.exp(z))
       else:
           return 1/(1+np.exp(-z))


    def sigmoid_derivative(self,z):
        return z*(1-z)


    def cross_entropy_error(self, PredictVector, TargetVector):
        Logits = np.log(PredictVector)
        Values = np.dot(TargetVector.T, Logits)

        return (np.sum(Values))

    def cost(self,y, t):
         return -np.sum(t * np.log(y))

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def softmax_stable(self,z):
        shiftz = z - np.max(z)
        exps = np.exp(shiftz)
        return exps / np.sum(exps)


    def cross_entropy_loss(self, pred_out, true_out):
        return -(np.sum((np.log2(pred_out) * true_out)))

    def cross_entropy(self, pred_out, true_out):
        return -(np.sum((np.log2(pred_out) * true_out), axis = 1))

    def cross_entropy_delta(self, pred_out, true_out):
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
        for layer_index in range(self.nlayers-1,0,-1):
            del_bias[layer_index] = deltas[layer_index]
            del_weights[layer_index] = np.dot(activation_outputs[layer_index-1].T, deltas[layer_index])
        return del_weights, del_bias


    def learn(self, training_inputs,training_outputs,learning_rate):

        categorical_tranining_outputs = self.to_categorical(training_outputs)
        estimated_output,activation_outputs = self.foreward_propagate(training_inputs)
        error = self.cross_entropy_delta(estimated_output,categorical_tranining_outputs)
        total_error = np.sum(error)
        del_weights, del_bias = self.backpropagate(error,activation_outputs)
        final_weights = {}
        final_bias = {}
        for layer in range(1,self.nlayers):
            final_weights[layer] = self.weights[layer-1] - learning_rate * del_weights[layer]
            final_bias[layer] = self.biases[layer-1] - learning_rate * (np.mean(del_bias[layer], axis=0)).reshape(self.biases[layer-1].shape[0],1)

        return final_weights, final_bias, total_error

    def train(self, training_inputs, training_outputs, epochs, learning_rate, batch_size ):

        costs = []
        for epoch in range(epochs):
            total_error = 0
            num_batches = int(len(training_inputs)/batch_size)
            start_of_batch = 0
            end_of_batch = len(training_outputs)
            for batch in range(num_batches):
                end_of_batch = start_of_batch + batch_size
                training_inputs_batch = training_inputs[start_of_batch:end_of_batch][:]
                training_outputs_batch = training_outputs[start_of_batch:end_of_batch][:]
                final_weights,final_bias,total_error_each = self.learn(training_inputs_batch,training_outputs_batch,learning_rate)
                total_error = total_error_each + total_error
                costs.append(total_error)
                final_weights_layer_list = list(final_weights.keys())
                final_bias_layer_list = list(final_bias.keys())
                for layer_index in final_weights_layer_list:
                    self.weights[layer_index-1] = final_weights[layer_index]

                for layer_index in final_bias_layer_list:
                    self.biases[layer_index-1] = final_bias[layer_index]

                start_of_batch = end_of_batch

            print(f"total error at epoch {epoch+1} is {total_error}")

        return self.weights,self.biases, costs


if __name__ == '__main__':

    x_train, y_train = mnist.train_images(), mnist.train_labels()
    nfeatures = x_train.shape[1] * x_train.shape[2]
    x_train_flatten = x_train.reshape((x_train.shape[0], nfeatures)).astype(np.float64)
    y_train = y_train.reshape((y_train.shape[0],1))
    x_test, y_test = mnist.test_images(),mnist.test_labels()
    x_test_flatten = x_test.reshape((x_test.shape[0],nfeatures)).astype(np.float64)
    y_test = y_test.reshape((y_test.shape[0], 1))
    epochs = 40
    learning_rate = 0.5
    n1_hidden = 100
    n_output = 10
    batch_size = 300
    layers = [nfeatures,n1_hidden,n_output]
    neural_nework1 = Neural_Network(layers)
    trained_weights, trained_biases, total_error_list = neural_nework1.train(x_train_flatten, y_train, epochs, learning_rate, batch_size)
    X = range(0, 2000, 500)
    plt.plot(X, total_error_list)
    plt.xlabel("No of epoch")
    plt.ylabel("Error")
    plt.legend("Error vs epoch")
    plt.show()