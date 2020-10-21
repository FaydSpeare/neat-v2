import neat
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
from hyperneat import network, geometry, substrate
import pickle


def create_data(size, num_samples, big_kernel_size, small_kernel_size):

    assert size % 9 == 0

    x, y = list(), list()
    x_set = set()

    big_kernel = math.floor(big_kernel_size / 2)
    small_kernel = math.floor(small_kernel_size / 2)

    while len(x) < num_samples:

        i, j = np.random.randint(big_kernel, size - big_kernel, 2)

        # Retry 10 times
        for _ in range(10):

            i2, j2 = np.random.randint(small_kernel, size - small_kernel, 2)

            # We don't want duplicates in the sample set
            if (i, j, i2, j2) not in x_set:

                # There must be a gap of 3 pixel between the centres
                # of the small and large squares
                gap = 2 + big_kernel + small_kernel
                if abs(i - i2) >= gap or abs(j - j2) >= gap:

                    sample = np.zeros((size, size))

                    # large square
                    for a in range(i - big_kernel, i + big_kernel + 1):
                        for b in range(j - big_kernel, j + big_kernel + 1):
                            sample[a, b] = 9. / big_kernel_size ** 2

                    # small square
                    for a in range(i2 - small_kernel, i2 + small_kernel + 1):
                        for b in range(j2 - small_kernel, j2 + small_kernel + 1):
                            sample[a, b] = 1. / small_kernel_size ** 2

                    x.append(sample)
                    y.append([i, j])
                    x_set.add((i, j, i2, j2))
                    break


    input_shape = output_shape = (size, size)

    input_substrate = geometry.rectangle(input_shape)
    output_substrate = geometry.rectangle(output_shape)
    cppn_inputs = substrate.create_sandwich_substrate(input_substrate, output_substrate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(len(output_substrate), use_bias=False, activation='linear'))
    model.add(tf.keras.layers.Reshape(output_shape))
    model.build((None, *input_shape))

    return tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y))), model, cppn_inputs


def get_error(model, data):

    for x, y in data.batch(len(data)):
        predictions = model(x).numpy()
    labels = y.numpy()

    error = 0
    distance = 0
    correct = 0
    size = predictions.shape[-1]

    for i, pred in enumerate(predictions):

        max_idx = np.argmax(pred)
        x_pred, y_pred = max_idx // size, max_idx % size
        x_label, y_label = labels[i]
        dx, dy = (2 / (size - 1)) * (x_pred - x_label), (2 / (size - 1)) * (y_pred - y_label)

        pred_error = dx ** 2 + dy ** 2
        error += pred_error
        distance += pred_error ** .5
        if pred_error == 0: correct += 1

    return error, distance, correct


DATA, model, substrate_sandwich = create_data(size=9, num_samples=500, big_kernel_size=3, small_kernel_size=1)


# We need to define an organism class that subclasses Organism
# Take a look at the Organism class in organism.py
class XOR(neat.Organism):

    def calculate_fitness(self):

        network.from_hyper_net(self, substrate_sandwich, model)
        error, distance, correct = get_error(model, DATA)

        self.correct = correct
        self.fitness =  -error
        self.distance = round(distance / len(DATA), 3)


# Note that we have access to information inside the organism objects, we could,
# for example, only terminate our search once the fitness reaches a certain threshold.
def fitness_assessor(organism):
    return organism.correct > len(DATA) - 1


if __name__ == '__main__':

    file_name = datetime.now().strftime('%d-%H-%M-%S')

    # Configure NEAT parameters
    custom_config = {
        'population_size' : 100,
        'num_inputs' : len(substrate_sandwich[0][0]),
        'num_outputs' : 1,

        'activations': ['gauss', 'sin', 'identity', 'sigmoid'],
        'activation_weights': [1, 1, 1, 1],
        'output_activation': 'identity',

        'compat_disjoint_coeff': 2.0,
        'compat_weight_coeff': 2.0,
        'compat_threshold': 2.5,

        'custom_print_fields': ['correct', 'distance'],
        'stats_file' : file_name + '.neat'
    }

    # Our subclassed organism type
    organism_type = XOR

    neat_solver = neat.Neat(organism_type, custom_config, assessor_function=fitness_assessor)

    # If we don't specify a number of generations, the search runs indefinitely
    neat_solver.run()

    # Retrieve the solvers
    solvers = neat_solver.get_solvers()

    organism = solvers[0]

    # Show prediction for 1 of the data points
    network.from_hyper_net(organism, substrate_sandwich, model)
    for x, y in DATA.batch(1):
        predictions = model(x).numpy()

    plt.imshow(x[0])
    plt.show()
    plt.imshow(predictions[0])
    plt.show()

    # Test the CPPN on a larger resolution (27 x 27)
    test_samples = 2
    test_data, scaled_model, scaled_substrate_sandwich = create_data(size=27, num_samples=test_samples, \
                                                                     big_kernel_size=9, small_kernel_size=3)
    network.from_hyper_net(organism, scaled_substrate_sandwich, scaled_model)

    # Show the predictions
    for x, y in test_data.batch(len(test_data)):
        predictions = scaled_model(x).numpy()

    for idx in range(test_samples):
        plt.imshow(x[idx])
        plt.show()
        plt.imshow(predictions[idx])
        plt.show()

    # Save the organism
    pickle.dump(organism, open(file_name + '.solver', 'wb'))


