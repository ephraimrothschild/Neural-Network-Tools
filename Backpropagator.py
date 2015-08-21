# Copyright (c) 2015 Ephraim Rothschild
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__author__ = 'Ephraim Rothschild'
import Generic_OVA
import random
import numpy as np
from sklearn.svm import NuSVC

# Class used for storing the data to input for the neural network
class Input:
    def __init__(self, x, y=None):
        # x is the input vector to be sent to the Neural Network. y is the label for that input vector.
        self.x = np.array(x, dtype=np.float64)
        self.y = y

# The class that uses backpropagation to predict a result based on an input vector
class BP:
    def __init__(self, k, d, maxitter=10, bias=None, v=None, W=None):
        # Parameters:
        #     k: the number of hidden units to use for the Neural Network
        #     d: The dimensionality of the input vectors to be used
        #     v: (Optional) The v vector weight. Only necessary if you are loading a saved set of weights.
        #     W: (Optional) The W weight matrix. Only necessary if you are loading a saved set of weights.
        #     maxitter: The number of times the train() method will loop when training the Neural Network.
        #       (100-1000 is recommended)

        self.maxitter = maxitter
        self.d = d
        self.k = k
        self.bias = np.float_(bias)

        # Initializes weights with random values between -1 and 1
        if W is None:
            self.W = np.float_((np.random.rand(k, d)*2 - 1)/10)
        else:
            self.W = W
        if v is None:
            self.v = np.float_((np.random.rand(k)*2 - 1)/10)
        else:
            self.v = v

    # Method for training the Neural Network
    def train(self, inputs, nu=0.01):
        # Parameters:
        #     inputs: An array of Input objects containing input vectors along with their corresponding labels.
        #     nu: The error rate. This will be multiplied by the gradient of the error function when subtracted from
        #       the weights. Value should be a very small number between 0 and 1 (ex: 0.01 or 0.001)

        for _ in range(0, self.maxitter):
            random.shuffle(inputs)
            # Loops through each of the inputs
            for input in inputs:
                # Append Bias if one is specified
                if self.bias: wb = np.append(input.x, self.bias)
                else: wb = input.x
                # Normalize input vectors for training
                x = wb/np.linalg.norm(wb)
                # a is equal to the weight matrix W multiplied by the input vector x
                a = self.W.dot(x)
                # h is equal to the vector containing hyperbolic tangent of each value in a
                h = np.tanh(a)
                # Find the error rate
                y_hat = np.tanh(np.dot(self.v, h))
                error = input.y - y_hat
                # Update v with the error*h proportional to nu
                self.v = self.v+nu*error*h
                for i in range(0, self.k):
                    # Update the weight vectors by subtracting the gradient of the error function
                    self.W[i] = self.W[i] + nu*((error*self.v[i])*(1 - (np.tanh(a[i])**2)))*x

    # Method for predicting a label given an Input containing an input vector
    def predict(self, input):
        # Parameters:
        #     input: An Input object containing an input vector to be used for predicting a label.

        if isinstance(input, Input):
            # Append a bias onto the input if one exists for the Neural Network
            if self.bias:
                wb = np.append(input.x, self.bias)
            else:
                wb = input.x
            # Normalize the input vector
            x = wb/np.linalg.norm(wb)
            h = np.tanh(self.W.dot(x))
        else:
            input_array = np.array(input, dtype=np.float64)
            # Append a bias onto the input if one exists for the Neural Network
            if self.bias:
                wb = np.append(input_array, self.bias)
            else:
                wb = input_array
            # Normalize the input vector
            x = wb/np.linalg.norm(wb)
            h = np.tanh(self.W.dot(x))
        return np.tanh(np.dot(h, self.v))

# Class that uses Scikit-Learn's implementation of SVM to predict labels
class svm():
    def __init__(self):
        # self.clf = SVC(kernel='rbf')
        self.clf = NuSVC()

    def train(self, inputs):
        # Parameters:
        #     inputs: An array of Input objects containing input vectors along with their corresponding labels.

        # Creates lists to use for fitting model
        X = []
        Y = []
        for data in inputs:
            X.append((data.x/np.linalg.norm(data.x)))
            Y.append(data.y)
        # Fit model
        self.clf.fit(X, Y)

    def predict(self, input):
        # Parameters:
        #     input: An Input object containing an input vector to be used for predicting a label.

        x = input.x/np.linalg.norm(input.x)
        if isinstance(input, Input):
            return self.clf.predict(x)
        else:
            x = input/np.linalg.norm(input)
            return self.clf.predict(x)

# Classifier that uses a 'One vs All' classification method.
# Can store any kind of predictor as long as that predictor has a predict(x) method
class OVAClassifier:
    def __init__(self, k, d, maxitter, nu=0.001):
        # Parameters:
        #     k: the number of hidden units to use for the Neural Networks
        #     d: The dimensionality of the input vectors to be used
        #     maxitter: The number of times the train() method will loop when training the Neural Network.
        #       (100-1000 is recommended)

        self.k = k
        self.d = d
        self.maxitter = maxitter
        self.classes = Generic_OVA.OVA()
        self.nu = nu

    def add_class_from_inputs(self, inputs, label_to_classify):
        # Parameters:
        #     inputs: an array of Input objects each containing an input vector and label
        #     label_to_classify: The label representing the "true" value for the given inputs. (ex: if you are
        #       trying to classify what a car looks like for the given input vectors, the value of this parameter
        #       should be something like "car".

        backprop_class = BP(self.k, self.d, self.maxitter)
        backprop_class.train(inputs, self.nu)
        self.classes.add_predictor(backprop_class, label_to_classify)

    def add_class_from_predictor(self, predictor, label_to_classify):
        # Parameters:
        #     predictor: The BP object to be used for training and prediction
        #     label_to_classify: The label representing the "true" value for the given inputs. (ex: if you are
        #       trying to classify what a car looks like for the given input vectors, the value of this parameter
        #       should be something like "car".

        self.classes.add_predictor(predictor, label_to_classify)

    def add_svm_class_from_inputs(self, inputs, label_to_classify):
        # Parameters:
        #     inputs: an array of Input objects each containing an input vector and label
        #     label_to_classify: The label representing the "true" value for the given inputs. (ex: if you are
        #       trying to classify what a car looks like for the given input vectors, the value of this parameter
        #       should be something like "car".

        support_vector = svm()
        support_vector.train(inputs)
        self.classes.add_predictor(inputs, label_to_classify)

    def predict(self, input):
        # Parameters:
        #     input: This can either be an Input object containing an input vector of size k, or just a numpy array
        #       of size k to be used as the input vector.
        #
        # Returns:
        #     The predicted label given the input vector.

        if isinstance(input, Input):
            return self.classes.predict(input.x)
        else:
            return self.classes.predict(input)

    def get_ova_result(self, x):
        # Parameters:
        #     x: the input that you would like to predict on
        #
        # Returns:
        #     An array representing the probabilities of the input vector being labeled for each of the labels.
        #         This is sorted in using the default sorting method on the labels as keys

        prob = self.classes.getProbabilities(x)
        return [value for (key, value) in sorted(prob.items())]

class MultiLayerClassifier:
    def __init__(self, k, d, maxitter, nu=0.01, layers=1):
        print("Number of layers: ", layers)
        # Parameters:
        #     k: the number of hidden units to use for the Neural Networks
        #     d: The dimensionality of the input vectors to be used
        #     maxitter: The number of times the train() method will loop when training the Neural Network.
        #       (100-1000 is recommended)
        #     nu: The error rate. This will be multiplied by the gradient of the error function when subtracted from
        #       the weights. Value should be a very small number between 0 and 1 (ex: 0.01 or 0.001)
        #     layers: The number of layer you want the neural network to use. The more layers, the more accurate the
        #         neural network will be, but the slower it will be to train. Since each layer is using a 2-layer
        #         backpropagation algorithm as its classifier, the number of true layers is actually the number of
        #         layers passed in for this parameter multiplied by 2. The default is 2, which is 4 true layers.

        self.k = k
        self.d = d
        self.maxitter = maxitter
        self.layers = layers
        self.inputs = {}
        self.nu = nu
        self.classifier = OVAClassifier(k, d, maxitter)

    def retrain(self):
        print("Retraining layer", self.layers)
        # Reset class's classifier
        self.classifier = OVAClassifier(self.k, self.d, self.maxitter, self.nu)
        # Create sub-layer's classifier using recursion
        if self.layers > 0:
            # Sets sub-layer to have the same k value, but d is equal to the number of classes, and the number of
            # layers is one less then the current layer's value.
            self.nextLayer = MultiLayerClassifier(self.k, len(self.inputs), self.maxitter, layers=self.layers-1)
        else:
            self.nextLayer = None
        if self.nextLayer:
            # Reset the inputs of the next layer
            self.nextLayer.inputs = {}
        # go through each of the inputs' keys
        for key in self.inputs.keys():
            # Create an array for positive inputs
            positive_inputs_for_class = []
            # Create an array for negative inputs
            negative_inputs_for_class = []
            # Create array that will hold both types of inputs
            inputs_for_class = []
            # Go through each of the various labels for our input vectors
            for other_key in self.inputs.keys():
                # Find out if this label is the same is the outer loop above
                if other_key == key:
                    # If it is, go through each of the input vectors with that label, and
                    # append it to positive_inputs_for_class.
                    for inner_value in self.inputs[other_key]:
                        positive_inputs_for_class.append(Input(inner_value, 1))
                else:
                    # If it isn't, go through each of the input vectors with that label, and
                    # append it to negative_inputs_for_class.
                    for inner_value in self.inputs[other_key]:
                        negative_inputs_for_class.append(Input(inner_value, -1))
            # Shuffle our arrays containing the positive and negative input vectors
            random.shuffle(positive_inputs_for_class)
            random.shuffle(negative_inputs_for_class)
            # From 0 to the length of the smallest array between positive_inputs_for_class,
            # and negative_inputs_for_class
            for i in range(0, min(len(positive_inputs_for_class), len(negative_inputs_for_class))):
                # Append one positive and one negative input vector to inputs_for_class
                inputs_for_class.append(positive_inputs_for_class[i])
                inputs_for_class.append(negative_inputs_for_class[i])
            # Train a neural net using the input vectors we just collected, to find the label from the outer loop above.
            print("Adding class " + str(key) + " to layer", self.layers)
            self.classifier.add_class_from_inputs(inputs_for_class, key)
        # If there is a next layer:
        if self.nextLayer:
            # Go through of our input vectors
            for key, value in self.inputs.items():
                for inner_value in value:
                    # Add the ova-result from predicting the given input vector - as an input vector itself to the
                    # next layer, with the label given by the above input vector's correct label.
                    self.nextLayer.add_input_for_class(np.array(self.classifier.get_ova_result(inner_value)), key)
            # Recurse
            self.nextLayer.retrain()

    def add_input_for_class(self, x, label):
        # Parameters:
        #     x: The input vector to be added as training data
        #     label: The label that you want to give to your training data

        if not (label in self.inputs):
            self.inputs[label] = []
        if isinstance(x, Input):
            self.inputs[label].append(x.x)
        else:
            self.inputs[label].append(x)

    def add_inputs_for_class(self, inputs, label):
        for input in inputs:
            self.add_input_for_class(input, label)

    def train(self, inputs):
        # Parameters:
        #     inputs: An array of Input objects containing input vectors along with their corresponding labels.

        for input in inputs:
            self.add_input_for_class(input.x, input.y)
        self.retrain()

    def predict(self, input):
        # Parameters:
        #     x: This can either be an Input object containing an input vector of size k, or just a numpy array
        #       of size k to be used as the input vector.

        if isinstance(input, Input):
            x = input.x
        else:
            x = input
        if self.nextLayer:
            prop = self.classifier.get_ova_result(x)
            return self.nextLayer.predict(np.array(prop))
        else:
            return self.classifier.predict(x)
