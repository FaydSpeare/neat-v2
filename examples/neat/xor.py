import neat
import numpy as np

# Data for XOR (inputs, labels)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([
    0.,
    1.,
    1.,
    0.
])

# We need to define an organism class that subclasses Organism
# Take a look at the Organism class in organism.py
class XOR(neat.Organism):

    def calculate_fitness(self):
        a = self.think(X)
        error = np.sum(abs(a - Y))
        correct = len(X) - error
        self.fitness = correct ** 2


# Optionally, we can define an assessor function. At each generation, after
# fitness is calculated for each organism, each organism is assessed using the
# assessor function we define. If the assessor function returns True, the NEAT
# search will terminate. A list of organism which 'solve' our assessor function
# can be retrieve from the NEAT object via the get_solvers() method.
def score_assessor(organism):
    a = organism.think(X)
    score = np.sum(abs(Y - a) < 0.5)
    return score == len(X)


# Note that we have access to information inside the organism objects, we could,
# for example, only terminate our search once the fitness reaches a certain threshold.
def fitness_assessor(organism):
    return organism.fitness > 15


if __name__ == '__main__':

    # Configure NEAT parameters
    custom_config = {
        'population_size' : 500,
        'num_inputs' : 2,
        'num_outputs' : 1
    }

    # Our subclassed organism type
    organism_type = XOR

    neat_solver = neat.Neat(organism_type, custom_config, assessor_function=score_assessor)

    # If we don't specify a number of generations, the search runs indefinitely
    neat_solver.run(generations=200)

    # Retrieve the solvers
    solvers = neat_solver.get_solvers()

    if solvers:

        # Test out the solver by providing inputs to the think method
        organism = solvers[0]
        print("Solver Fitness:", organism.fitness)
        print("Solver outputs for XOR data:")
        outputs = organism.think(X)[0]
        for idx, output in enumerate(outputs):
            print(X[idx], '->', round(output, 3))

        # Now enjoy your trained network!




