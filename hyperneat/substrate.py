def create_sandwich_substrate(input_substrate, output_substrate):
    inputs = []
    for o, (ox, oy) in enumerate(input_substrate):
        row_inputs = []
        for i, (ix, iy) in enumerate(output_substrate):
            dx = ox - ix
            dy = oy - iy
            #dxdy = (dx**2 + dy**2)**(1/2)
            hyper_input = [ix, iy, ox, oy, dx, dy, 1.]
            row_inputs.append(hyper_input)
        inputs.append(row_inputs)
    return inputs