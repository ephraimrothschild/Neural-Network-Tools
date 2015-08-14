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
from operator import itemgetter

class OVA:
    def __init__(self):
        self.predictor_wrappers = []

    # Adds a predictor to the OVA Object. The predictor must have a .predict(x) method
    def add_predictor(self, network, label_to_classify):
        # Parameters:
        #     network: The Neural Network or predictor object to add to the OVA.
        #              The predictor must have a .predict(x) method.
        #     label_to_classify: The label to use for the class represented by this predictor

        self.predictor_wrappers.append(PredictorWrapper(network, label_to_classify))

    def predict(self, x):
        # Parameters:
        #     x: The input vector. The way this vector is represented is dependent on the parameter type of the
        #        added predictors. Generally a numpy array is used.
        #
        # Returns: The predicted label for the given input vector

        probability = self.getProbabilities(x)
        sorted_probability = sorted(probability.items(), key=itemgetter(1), reverse=True)
        return sorted_probability[0][0]

    def getProbabilities(self, x):
        # Parameters:
        #     x: The input vector. The way this vector is represented is dependent on the parameter type of the
        #        added predictors. Generally a numpy array is used.
        #
        # Returns: a dictionary containing names of classes as keys, and the corresponding likelihood of the given
        #          input vector belonging to that class.

        probability = {}
        for predictor_wrapper in self.predictor_wrappers:
            prediction_certainty = predictor_wrapper.predictor.predict(x)
            probability[predictor_wrapper.label_to_classify] = prediction_certainty
        return probability

class PredictorWrapper:
    def __init__(self, predictor, label_to_classify):
        self.predictor = predictor
        self.label_to_classify = label_to_classify