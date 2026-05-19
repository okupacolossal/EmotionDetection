import numpy as np
filters = np.random.randn(32, 3, 3) * 0.01
biases = np.zeros(32)

flat_size = 32 * 23 * 23   # 16928
W1 = np.random.randn(flat_size, 128) * 0.01
b1 = np.zeros(128)

W2 = np.random.randn(128, 5) * 0.01
b2 = np.zeros(5)

def relu(x):
    return np.maximum(x, 0)
        
def conv_forward(input, filters, biases):
    num_filters = filters.shape[0]
    h_out = input.shape[0] - 3 + 1
    w_out = input.shape[1] - 3 + 1
    output = np.zeros((num_filters, h_out, w_out))

    for f_idx, filter in enumerate(filters):
        for row in range(h_out):
            for col in range(w_out):
                patch = input[row:row+3, col:col+3]
                output[f_idx, row, col] = np.sum(patch * filter) + biases[f_idx]
    return output

def maxpool_forward(input, pool_size = 2):

    num_filters, h, w = input.shape
    h_out = h // 2
    w_out = w // 2
    output = np.zeros((num_filters, h_out, w_out))
    
    for filter in range(num_filters):
        for row in range(h_out):
            for col in range(w_out):
                patch = input[filter, row*pool_size:row*pool_size+pool_size, col*pool_size:col*pool_size+pool_size]
                output[filter, row, col] = np.max(patch)

    return output

def flatten(x):
    return x.reshape(-1)

def fc_forward(input, W, b):
    return np.dot(input, W) + b

def softmax(x):
    num = np.exp(x)
    return num / np.sum(num)

def cross_entropy(probs, label):
    return -np.log(np.sum(probs * label))


        

    
    




        
