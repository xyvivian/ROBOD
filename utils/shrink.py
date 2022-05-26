import numpy as np

"""
@Original Author: Prof. Randy
@Modified by: Chong Zhou
update to python3: 03/15/2019
Args:
    epsilon: the shrinkage parameter (either a scalar or a vector)
    x: the vector to shrink on
"""

def l21shrink(epsilon, x):
    """
    The proximal method to evaluate l12 norm introduced by Section 3.3
    from http://library.usc.edu.ph/ACM/KKD%202017/pdfs/p665.pdf (Also see section 4.3)
    """
    output = x.copy()
    norm = np.linalg.norm(x, ord=2, axis=0)
    for i in range(x.shape[1]):
        if norm[i] > epsilon:
            for j in range(x.shape[0]):
                output[j,i] = x[j,i] - epsilon * x[j,i] / norm[i]
        else:
            output[:,i] = 0.
    return output


def l1shrink(epsilon,x):
    """
    The proximal method to evaluate l1 norm  from
    http://library.usc.edu.ph/ACM/KKD%202017/pdfs/p665.pdf (Also see section 4.2)
    """
    output = np.copy(x)
    above_index = np.where(output > epsilon)
    below_index = np.where(output < -epsilon)
    between_index = np.where((output <= epsilon) & (output >= -epsilon))
    
    output[above_index[0], above_index[1]] -= epsilon
    output[below_index[0], below_index[1]] += epsilon
    output[between_index[0], between_index[1]] = 0
    return output