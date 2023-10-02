import tensorflow as tf
tf.config.run_functions_eagerly(True)



class NeuralNetworkTf(tf.keras.Sequential): #Define the class of the neural network.

  def __init__(self, sizes, random_state=1):
    
    super().__init__()
    self.sizes = sizes
    self.random_state = random_state
    tf.random.set_seed(random_state)

    #Mistake 1: We need to flatten the input before feeding it to the network since we are using a dense layer.
    self.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    
    for i in range(0, len(sizes)):

    #Mistake 2: We need to add a correct loop to add the layers to the network.
      if i == len(sizes) - 1:
        self.add(tf.keras.layers.Dense(sizes[i], activation='softmax')) #Mistake 3: We need to use softmax as the activation function of the last layer.
        
      else:
        self.add(tf.keras.layers.Dense(sizes[i], activation='sigmoid')) #Mistake 4: We need to use sigmoid as the activation function of the hidden layers.
        
  
  def compile_and_fit(self, x_train, y_train, x_val=None, y_val=None, #add validation data for extract the best model
                      epochs=50, learning_rate=0.01, #add epochs and learning rate
                      batch_size=1,validation_data=None): #add batch size and validation data
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    eval_metrics = ['accuracy']

    super().compile(optimizer=optimizer, loss=loss_function, 
                    metrics=eval_metrics)
    return super().fit(x_train, y_train, epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=validation_data)  



class TimeBasedLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
  '''TODO: Implement a time-based learning rate that takes as input a 
  positive integer (initial_learning_rate) and at each step reduces the
  learning rate by 1 until minimal learning rate of 1 is reached.
    '''

  def __init__(self, initial_learning_rate, decay = 1,  min_learning_rate=1):
        super(TimeBasedLearningRate, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay = tf.cast(decay, tf.float32)
        self.min_learning_rate = min_learning_rate

  def __call__(self, step):
     step = tf.cast(step, tf.float32)
     learning_rate = tf.maximum(self.min_learning_rate, self.initial_learning_rate - step * self.decay)
     return learning_rate



    