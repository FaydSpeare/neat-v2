import numpy as np

def square(size):
    return np.array([(i / (size - 1), j / (size - 1)) for i in range(size) for j in range(size)]).astype('float32')

def rectangle(shape):
    isize, jsize = shape
    return np.array([(2*(i / max(1, isize - 1))-1, 2*(j / max(1, jsize - 1))-1) for i in range(isize) for j in range(jsize)]).astype('float32')

def create_cppn_inputs(input_substrate, output_substrate):
    inputs = []
    for o, (ox, oy) in enumerate(input_substrate):
        row_inputs = []
        for i, (ix, iy) in enumerate(output_substrate):
            hyper_input = [ix, iy, ox, oy, 1.]
            row_inputs.append(hyper_input)
        inputs.append(row_inputs)
    return inputs