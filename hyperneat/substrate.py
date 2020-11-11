def create_sandwich_substrate(input_substrate, output_substrate):
    inputs = []
    for o, (ox, oy) in enumerate(input_substrate):
        row_inputs = []
        for i, (ix, iy) in enumerate(output_substrate):
            dx = ox - ix
            dy = oy - iy
            #hyper_input = [ix, ox, dx, iy, oy, dy, 1.]
            hyper_input = [dx, dy]
            row_inputs.append(hyper_input)
        inputs.append(row_inputs)
    return inputs