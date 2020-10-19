import neat
import tensorflow as tf
from hyperneat import network, substrate
import numpy as np
import random
import matplotlib.pyplot as plt

def create_test():
    size = 27
    point = np.zeros((size, size))

    a = random.randint(4, size - 5)
    b = random.randint(4, size - 5)

    for i in range(-4, 5):
        for j in range(-4, 5):
            point[a + i, b + j] = .25

    a = random.randint(1, size - 2)
    b = random.randint(1, size - 2)
    for i in range(-1, 2):
        for j in range(-1, 2):
            point[a + i, b + j] = .25

    input_shape = output_shape = (size, size)

    input_substrate = substrate.rectangle(input_shape)
    output_substrate = substrate.rectangle(output_shape)
    big_cppn_inputs = substrate.create_cppn_inputs(input_substrate, output_substrate)

    big_model = tf.keras.Sequential()
    big_model.add(tf.keras.layers.Flatten())
    big_model.add(tf.keras.layers.Dense(len(output_substrate), use_bias=False, activation='linear'))
    big_model.add(tf.keras.layers.Reshape(output_shape))
    big_model.build((None, *input_shape))

    return point, big_model, big_cppn_inputs



def create_data(size):

    x, y = list(), list()
    for i in range(1, size - 1):
        for j in range(1, size - 1):

            for n in range(1):

                point = np.zeros((size, size))

                point[i - 1, j - 1] = 1.
                point[i - 1, j] = 1.
                point[i - 1, j + 1] = 1.

                point[i, j - 1] = 1.
                point[i, j] = 1.
                point[i, j + 1] = 1.

                point[i + 1, j - 1] = 1.
                point[i + 1, j] = 1.
                point[i + 1, j + 1] = 1.

                a = random.randint(0, size - 1)
                b = random.randint(0, size - 1)
                point[a, b] = 1.

                x.append(point)
                y.append([i, j])

    input_shape = output_shape = (size, size)

    input_substrate = substrate.rectangle(input_shape)
    output_substrate = substrate.rectangle(output_shape)
    cppn_inputs = substrate.create_cppn_inputs(input_substrate, output_substrate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(len(output_substrate), use_bias=False, activation='linear'))
    model.add(tf.keras.layers.Reshape(output_shape))
    model.build((None, *input_shape))

    return tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y))), model, cppn_inputs


size = 9
DATA, model, cppn_training_inputs  = create_data(size)
print(len(DATA))


# We need to define an organism class that subclasses Organism
# Take a look at the Organism class in organism.py
class XOR(neat.Organism):

    def calculate_fitness(self):
        network.from_hyper_net(self, cppn_training_inputs, model)
        for x, y in DATA.batch(len(DATA)):
            out = model(x)

        outputs = out.numpy()
        ys = y.numpy()
        error = 0
        correct = 0
        size = out.shape[2]
        for idx, o in enumerate(outputs):
            argmax = np.argmax(o)
            i = argmax // size
            j = argmax % size
            error += (2 * (ys[idx, 0] - i)/size)**2 + (2 * (ys[idx, 1] - j)/size)**2
            if ys[idx, 0] == i and ys[idx, 1] == j:
                correct +=1


        #print(correct, error / len(DATA))

        #error = tf.keras.losses.mse(y, y_preds)
        m = model
        self.correct = correct
        self.fitness = -error
        #if self.fitness == -24.0:
        #   print()


# Note that we have access to information inside the organism objects, we could,
# for example, only terminate our search once the fitness reaches a certain threshold.
def fitness_assessor(organism):
    #return organism.fitness / len(DATA) > -0.01
    return organism.correct > (size - 2)**2 - 20


if __name__ == '__main__':


    # Configure NEAT parameters
    custom_config = {
        'population_size' : 100,
        'num_inputs' : 5,
        'num_outputs' : 1,

        'activations': ['gauss', 'sin', 'identity', 'sigmoid'],
        'activation_weights': [1, 1, 1, 1],
        'output_activation': 'identity',

        'compat_disjoint_coeff': 1.0,
        'compat_weight_coeff': 2.0,
        'compat_threshold': 3.0
    }

    # Our subclassed organism type
    organism_type = XOR

    neat_solver = neat.Neat(organism_type, custom_config, assessor_function=fitness_assessor)

    # If we don't specify a number of generations, the search runs indefinitely
    neat_solver.run(generations=1000)

    # Retrieve the solvers
    solvers = neat_solver.get_solvers()

    organism = solvers[0]

    network.from_hyper_net(organism, cppn_training_inputs, model)
    for x, y in DATA.batch(len(DATA)):
        out = model(x)

    outputs = out.numpy()
    ys = y.numpy()
    error = 0
    correct = 0
    size = out.shape[2]
    for idx, o in enumerate(outputs):
        argmax = np.argmax(o)
        i = argmax // size
        j = argmax % size
        error += (2 * (ys[idx, 0] - i) / size) ** 2 + (2 * (ys[idx, 1] - j) / size) ** 2
        if ys[idx, 0] == i and ys[idx, 1] == j:
            correct += 1

    print(correct, error / len(DATA))

    plt.imshow(outputs[0])
    plt.show()

    width = 10
    height = 10
    rows = 5
    cols = 5
    axes = []
    fig = plt.figure()

    for a in range(rows * cols):
        axes.append(fig.add_subplot(rows, cols, a + 1))
        subplot_title = ("Subplot" + str(a))
        axes[-1].set_title(subplot_title)
        plt.imshow(outputs[a] / np.max(outputs[a]))
    fig.tight_layout()
    plt.show()

    print()

    for i in range(10):
        t, bigm, bigcppn = create_test()
        network.from_hyper_net(organism, bigcppn, bigm)
        out = bigm(tf.constant([t])).numpy()[0]
        plt.imshow(t)
        plt.show()
        plt.imshow(out)
        plt.show()

    import pickle
    pickle.dump(organism, open('solver.pickle', 'wb'))


