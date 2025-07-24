import numpy as np


def round_to_array(object, array):
    """
    Round the elements of `obj` to the nearest value in `arr`.

    Parameters:
    obj (array-like): The object containing values to be rounded.
    arr (array-like): The 1-D array of values to round to.

    Returns:
    np.ndarray: An array with the same shape as `obj`, where each element is replaced by the closest value from `arr`.
    """
    object = np.asarray(object)
    array = np.asarray(array)
    if array.shape != (array.size,):
        raise ValueError("arr must be a 1-D array.")

    # Find the index in array of the closest value for each value in object
    indices = np.abs(np.expand_dims(object, -1) - array).argmin(axis=-1)

    # Replace each entry in object with the closest entry from array
    return array[indices]
