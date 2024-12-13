# TensorFlow_DeepLearning_CNN_CPU_GPU_Bencharmk_Test
## Tested Environment:
    Tensorflow 2.17
    Ryzen 3900
    GPU 3090
## Code Flow
    Load the MNST Image Data (to classify number 0 to 9)
    Define Deep learning model - CNN (Convolution Neuro Network) 
    Training the Model in CPU and GPU separately
    Compare the time used by CPU and GPU
## Training Process
    Forward Propagation: Data flows from the input to output layer, applying weights and activation functions.
    Loss Calculation: Compares predicted output with actual labels.Backpropagation: Adjusts weights to minimize loss.
    Optimization: Uses Adam optimizer to update weights and biases.
## Brief of Deep Learning Model Defined:
    Input Layer: 
        Shape: (28, 28, 1)
    Convolutional Layer: 
        32 filters, size 3x3, ReLU activation
    MaxPooling Layer:
        Pool size 2x2
    Convolutional Layer:
        64 filters, size 3x3, ReLU activation
    MaxPooling Layer:
        Pool size 2x2 
    Flatten Layer: 
        Converts to 1D vector
    Dense Layer:
        64 neurons, ReLU activation
    Output Layer:
        10 neurons, Softmax activation