# Neural-Network-Tools
A generic implementation of a Neural Network using a backpropagation algorithm that can be used for any kind for vector-represented data.


### Classes:
 - `Backpropagator.Input` - The class used for feeding training data to the neural network.  
    Instantiate an `Input` object with:  
    
     ```python
     Input(input_vector_as_numpy-array, label)
     ```
 - `Backpropagator.BP` - The class storing the Neural Network.
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