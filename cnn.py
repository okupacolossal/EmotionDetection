import numpy as np

class EmotionCNN:
    def __init__(self, image_size=48):
        
        self.filter_size = 3
        self.num_filters = 32
        self.conv_filters = np.random.randn(self.num_filters, self.filter_size, self.filter_size) * 0.01
        # This is our convolutional layer, for our CNN. Why is it 32, 3 3?
        # We want 32 filters sliding accross an image, 3*3 in size.
        # this creates exactly that, a list of 32 values, in 3x3 formats, with random
        # initialization. This is the filter idea

        self.conv_biases = np.zeros(32)
        # This is the bias for each filter, we have 32 filters, so we have 32 biases. The weighted sum
        # will apply for each filter, so we need a bias for each one aswell. We initiate these as zeros

        self.max_pool_size = 2
        # Max pooling is a downsampling technique, it reduces the spatial dimensions of the input. 
        # It does this by taking the maximum value from a set of values in a defined window (in this case, 2x2). 
        # This helps to reduce the computational complexity and also helps 
        # to make the model more robust to small translations in the input. 

        self.pos_size = (image_size - 3 + 1) // 2
        # After the convolutional layer, the spatial dimensions of the output will be reduced. 
        # The formula for calculating the output size after convolution is:
        # output_size = (input_size - filter_size + 2 * padding) / stride + 1
        # In our case, we have an input size of 48, a filter size of 3, no padding, and a stride of 1.
        # So the output size after convolution will be:
        # output_size = (48 - 3 + 0) / 1 + 1 = 46
        # After max pooling with a pool size of 2, the spatial dimensions will be halved, so the final output size will be:
        # pos_size = 46 // 2 = 23

        self.hidden_size = 128
        # This is the number of neurons in the hidden layer of our fully connected network.
        
        W1, b1 = np.random.randn((self.pos_size * self.pos_size * self.num_filters), self.hidden_size)
        

        self.fc1 = None
        self.fc2 = None
        self.max_pool = None
        self.conv_layer = None

    def forward(self):
        pass

    
Cnn = EmotionCNN()




        
