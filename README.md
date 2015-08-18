# Neural-Network-Tools
A generic implementation of a Neural Network using a backpropagation algorithm that can be used for any kind for vector-represented data.

### Requirements:
   There are a few things you are going to have to install before using this:
   1. [Python 3.x][2]
   2. [Numpy/SciPy][3]
   3. [Scikit-Learn][4] - This is only used for the svm() class. The BP class runs without this.
  
### Classes:
 - `Backpropagator.Input` - The class used for feeding training data to the neural network.  
    Instantiate an `Input` object with:  
    
     ```python
     Input(input_vector_as_numpy-array, label)
     ```
 - `Backpropagator.BP` - The class containing the Neural Network.
    A `BP` object is instantiated with the statement:
    
      ```python
      BP(num_of_hidden_units, dimensions, num_loops, bias)
      ```

    If you want to create a `BP` object based on existing weights, use:  
    
      ```python
      BP(num_of_hidden_units, dimensions, num_loops, bias, v_vector, W_weight_matrix)
      ```
      
 - `Backpropagator.OVAClassifier` - A class used for implementing a One-vs-all multi-class classification algorithm.  
 	An `OVAClassifier` is instantiated with the statement:
 	  
      ```python
      Backpropagator.OVAClassifier(hidden_units, dimensions, num_loops, gradient_corrective_step)
      ```
     
 - `Generic_OVA.OVA` - A class containing a generic implementation of a One-vs-All classifier. Unlike `OVAClassifier`, 
    this is intended to be used if you have written your own predictor from scratch.
    An `OVA` object is instantiated with the statement:

      ```python
      Generic_OVA.OVA()
      ```

  
### Building your Neural Network:
Any great README should have a good exmple of how to use the tools. So here is one that I wrote to train a neural network    for recognizing numbers using the [USPS Dataset][1].

```python
    import random
    import io
    import numpy as np
    import Backpropagator

    def getInputs(path, num_to_classify):
        training_file = open(path, "r")
        raw_training_data = np.loadtxt(training_file).tolist()
        random.shuffle(raw_training_data)
        num_data = []
        not_num_data = []
        training_data = []
        for data in raw_training_data:
            training_array = data[1:]
            training_label = data[0]
            if training_label == num_to_classify:
                num_data.append(Backpropagator.Input(training_array, 1))
            else:
                not_num_data.append(Backpropagator.Input(training_array, -1))
        for num in range(0,len(num_data)):
            training_data.append(num_data[num])
            training_data.append(not_num_data[num])
        return training_data


    def start():
        ova = Backpropagator.OVAClassifier(15, 257, 20, nu=0.1)
        for num in range(0, 10):
            training_data = getInputs("usps.train", num)
            backprop = Backpropagator.BP(15, 257, 20, bias=1)
            backprop.train(training_data, 0.2)
            ova.add_class_from_predictor(backprop, num)
        test(ova)


    def test(multiclass):
        test_file = open("usps.test", "r")
        text_lines = test_file.readlines()
        lines = []
        for num in range(0, 1000):
            lines.append(np.loadtxt(io.StringIO(text_lines[num])))
        true = 0
        false = 0
        predictions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for num in range(0, 1000):
            line = lines[num]
            inVec = line[1:]
            result = multiclass.predict(np.array(inVec))
            predictions[int(result)] += 1
            labels[int(line[0])] += 1
            if result == line[0]:
                true += 1
            else:
                false += 1
        for index in range(0, 10):
            print("Number of " + index.__str__() + "s predicted:   " + predictions[index].__str__())
            print("                     Vs:  "+ labels[index].__str__() + " real ones")
        print()
        print("Number of Accurate Estimates:  " + true.__str__())
        print("Number of Errors:              " + false.__str__())

    start()
```
And just in case you were wondering about the accuracy of the algorithm, here is the output produced by the above code:

```
Number of 0s predicted:   203
                     Vs:  199 real ones
Number of 1s predicted:   119
                     Vs:  120 real ones
Number of 2s predicted:   101
                     Vs:  109 real ones
Number of 3s predicted:   80
                     Vs:  81 real ones
Number of 4s predicted:   76
                     Vs:  93 real ones
Number of 5s predicted:   60
                     Vs:  53 real ones
Number of 6s predicted:   104
                     Vs:  101 real ones
Number of 7s predicted:   61
                     Vs:  60 real ones
Number of 8s predicted:   93
                     Vs:  95 real ones
Number of 9s predicted:   103
                     Vs:  89 real ones

Number of Accurate Estimates:  913
Number of Errors:              87
```

[1]: http://www.mathworks.com/matlabcentral/fileexchange/48567-usps-digit-dataset
[2]: https://www.python.org/downloads/
[3]: http://www.scipy.org/install.html
[4]: http://scikit-learn.org/stable/
