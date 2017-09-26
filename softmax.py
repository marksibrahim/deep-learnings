

def softmax(x):
    """
    Compute the softmax of a vector/matrix using numpy

    The max from each row is substracted for numerical stability,
    since np.exp() overflows for values greater than ~700
    https://nolanbconaway.github.io/blog/2017/softmax-numpy

    Args:
        x (numpy array): n-dimensional vector or mxn array
    Returns:
        x (numpy array): with softmax applied (same dimensions)
    """
    if len(x.shape) > 1:
        # Matrix
        # substracting max leaves function unchanged due to softmax's invariance to sums by a constant 
        # keepdims= True, because broadcasting requires trailing shape entries to match
        x -= np.max(x, axis=1, keepdims=True)
        x = np.exp(x)
        sum_exp_xj = np.sum(x, axis=1, keepdims=True)
        x = np.divide(x, sum_exp_xj)
    else:
        # Vector
        x -= np.max(x)
        x = np.exp(x)
        sum_exp_xj = np.sum(x)
        x = np.divide(x, sum_exp_xj)
    return x
