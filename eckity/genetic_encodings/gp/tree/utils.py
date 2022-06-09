"""
This module implements some utility functions. 
"""


def create_terminal_set(X):
    """
    Create a terminal set from a 2D-shaped numpy array. 

    Example: \n
        X = array([[  4,   7,  -7, -10],  \n
                   [  7,  -3,   3,  -8],  \n
                   [  8,  -5,  -3,  -1]])

    Returns the list: \n
        ['x0', 'x1', 'x2', 'x3']

    Parameters
    ----------
    X : 2d numpy array
        The array from which we wish to extract features -- which will become terminals.

    Returns
    -------
    Terminal set (a list).

    """
    
    return [f'x{i}' for i in range(X.shape[1])]


def _generate_args(X):
    """
    Generate keyword arguments from a 2d array for passing to GPTree.execute.
    
    Example: \n 
        X = array([[  4,   7,  -7, -10],  \n
                   [  7,  -3,   3,  -8],  \n
                   [  8,  -5,  -3,  -1]])

    Returns the dict: \n
        {'x0': array([4, 7, 8]), \n
         'x1': array([ 7, -3, -5]), \n
         'x2': array([-7,  3, -3]), \n
         'x3': array([-10,  -8,  -1])}     

    Returns
    -------
    kwargs : dict
        A keyword dictionary that includes a value for every variable x_i in the terminal set. 

    """
    
    kwargs = dict()
    for i in range(X.shape[1]):
        kwargs[f'x{i}'] = X[:, i]
    return kwargs
