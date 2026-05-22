import numpy as np

class EmotionCNN:
    def __init__(self, image_size=48):

        self.image_size = 48
        
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

        # First layer's weights & biases
        self.W1 = np.random.randn((self.pos_size * self.pos_size * self.num_filters), self.hidden_size) * 0.01
        self.b1 = np.zeros(self.hidden_size)
        
        # Second layer's weights & biases
        self.W2 = np.random.randn(128, 5) * 0.01
        self.b2 = np.zeros(5)

    def forward(self, x):

        self.input = x
        # save x as the input

        result = self.conv_forward(x)
        result = self.relu(result)
        self.conv_output = result
        # save conv layer + relu output 

        result = self.maxpool_forward(result)
        self.pool_shape = result.shape

        result = self.flatten(result)
        self.flat = result
        # store flatten

        result = self.fc_forward(result, self.W1, self.b1)
        result = self.relu(result)
        self.fc1_out = result
        # store fc1 output
        
        result = self.fc_forward(result, self.W2, self.b2)
        result = self.softmax(result)
        
  
        return result
    
    def backward(self, predicted, true):
        
        # true is our OHE label for the true value of whatever emotion we were trying to detect
        # [0, 0, 1, 0, 0], for example

        # predicted is what our model's predictons
        # [0.1, -0.3, 0.5, 0.2, -0.7], for example

        dL = predicted - true
        # dL is the loss function, predicted - true
       
        dW2 = self.fc1_out.reshape(-1,1) @ dL.reshape(1,-1)
        # @ is the symbol for matrix multiplication, and matrix multiplication requires the shape to match
        # the shape of fc1_out is (128,) (128 neurons in the hidden layer) the shape of dL (which is our probabilities is 5,)
        # so doing this reshape, we make fc1_out (128, 1) and dL (1, 5), so the result of the matrix multiplication is (128, 5)
        # which is the shape of W2

        db2 = dL
        # db2 (hidden layer 2's bias) is the value of dL which is the loss function
        # why do we ues it as a bias? because during backprop, we ask, "what caused this error?"; the bias contributed to the error
        # by exactly the amount of the error itself, because it's a flat addition with no multiplication~

        dL = dL @ self.W2.T
        # To unpack; .T means transpose, in the forward pass we went 128 -> 5, in the backward pass we go 5- -> 128
        # W2 (3, 2):          W2.T (2, 3):
        # [[0.5, 0.1],        [[0.5, 0.3, 0.2],
        #  [0.3, 0.8],   →    [0.1, 0.8, 0.4]]
        #  [0.2, 0.4]]
        # rows become columns, columns become rows
        
        # dL @ W2.T:
        # [-0.3, 0.6] @ [[0.5, 0.3, 0.2],
        #                [0.1, 0.8, 0.4]]
        #
        # neuron 0: (-0.3 × 0.5) + (0.6 × 0.1) = -0.15 + 0.06 = -0.09
        # neuron 1: (-0.3 × 0.3) + (0.6 × 0.8) = -0.09 + 0.48 =  0.39
        # neuron 2: (-0.3 × 0.2) + (0.6 × 0.4) = -0.06 + 0.24 =  0.18
        #
        # result = [-0.09, 0.39, 0.18]
        # each neuron receives blame proportional to its weight connections

        dL = dL * (self.fc1_out > 0)
        # relu backprop — passes the gradient through where fc1_out was active (> 0)
        # blocks the gradient (sets to 0) where the neuron was dead during forward pass
    
        dW1 = self.flat.reshape(-1,1) @ dL.reshape(1,-1)
        db1 = dL
        dL = dL @ self.W1.T
        # same as dW2

        
        dL = dL.reshape(self.pool_shape)


        # de-pool
        conv = self.conv_output
        empty_conv = np.zeros(conv.shape)

        for index in range(self.num_filters):
            for row in range(23):
                for col in range(23):
                    empty_conv[index, row*2:row*2+2, col*2:col*2+2] += dL[index, row, col] * self.max_mask[index, row*2:row*2+2, col*2:col*2+2]

        dL = empty_conv
        dL = dL * (self.conv_output > 0)

        dW_conv = np.zeros_like(self.conv_filters)
        db_conv = np.zeros_like(self.conv_biases)

        size_after = self.image_size - self.filter_size + 1 

        for index, filter in enumerate(self.conv_filters):
            for col in range(size_after):
                for row in range(size_after):
                    patch = self.input[row:row+3, col:col+3]
                    dW_conv[index] += patch * dL[index, row, col]
                    db_conv[index] += dL[index, row, col]

        return dW_conv, db_conv, dW1, db1, dW2, db2
    
    def fc_forward(self, x, W, b):
        return x @ W + b
    
    def softmax(self, x):
        exp = np.exp(x)
        return exp / np.sum(exp)
            
    def conv_forward(self, x): 

        size_after = self.image_size - 3 + 1
        output_array = np.zeros((self.num_filters, size_after, size_after))

        for index, filter in enumerate(self.conv_filters):
            for col in range(size_after):
                for row in range(size_after):
                    patch = x[row:row+3, col:col+3]
                    output_array[index, row, col] = np.sum(patch * filter) + self.conv_biases[index]
        
        # loop over the filter, then by col and row, get 3:3 patches (overlapping), sum up whats in the patch n filter and add the bias
        
        return output_array
    
    def maxpool_forward(self, x):

        size_after = (self.image_size - 3 + 1) // 2
        output_array = np.zeros((self.num_filters, size_after, size_after))
        self.max_mask = np.zeros_like(x)
        
        for index in range(self.num_filters):
            for col in range(size_after):
                for row in range(size_after):
                    patch = x[index, row * 2:row * 2 + 2, col * 2:col*2+2]
                
                    highest_value = np.max(patch)

                    pos = np.unravel_index(np.argmax(patch), patch.shape)
                    local_row, local_col = pos

                    self.max_mask[index, row*2 + local_row, col*2 + local_col] = 1

                    output_array[index, row, col] = highest_value
        
        return output_array
 
    def relu(self, x):
        return np.maximum(0, x)
    
    def flatten(self, x):
        return x.reshape(-1)

    def update(self, dW_conv, db_conv, dW1, db1, dW2, db2, lr=0.01):
        self.W1 -= dW1 * lr
        self.b1 -= db1 * lr
        self.W2 -= dW2 * lr
        self.b2 -= db2 * lr
        self.conv_filters -= dW_conv * lr
        self.conv_biases -= db_conv * lr


def train(cnn, X_train, y_train, epochs, lr=0.01, batch_size=32):
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        epoch_loss = 0 

        # split into batches
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            dW_conv_acc = np.zeros_like(cnn.conv_filters)
            db_conv_acc = np.zeros_like(cnn.conv_biases)
            dW1_acc = np.zeros_like(cnn.W1)
            db1_acc = np.zeros_like(cnn.b1)
            dW2_acc = np.zeros_like(cnn.W2)
            db2_acc = np.zeros_like(cnn.b2)

            if i % (batch_size * 2) == 0:
                    print(f"  Batch {i//batch_size}/{len(X_train)//batch_size}")

            for image, label in zip(X_batch, y_batch):

                predicted = cnn.forward(image)
                loss = -np.log(predicted[np.argmax(label)] + 1e-8)
                dW_conv, db_conv, dW1, db1, dW2, db2 = cnn.backward(predicted, label)
                dW_conv_acc += dW_conv
                db_conv_acc += db_conv
                dW1_acc += dW1
                db1_acc += db1
                dW2_acc += dW2
                db2_acc += db2
                epoch_loss += loss
                

            
            cnn.update(dW_conv_acc/batch_size, db_conv_acc/batch_size, 
                dW1_acc/batch_size, db1_acc/batch_size, 
                dW2_acc/batch_size, db2_acc/batch_size, lr)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(X_train):.4f}")
        
 


        
