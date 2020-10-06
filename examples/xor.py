from organism import Organism
from neat import Neat

# Data for XOR (inputs, labels)
DATA = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

# We need to define an organism class that subclasses Organism
# Take a look at the Organism class in organism.py
class XOR(Organism):

    def calculate_fitness(self):
        error = sum([abs(y - self.think(x)[0]) for x, y in DATA])
        correct = len(DATA) - error
        self.fitness = correct ** 2


# Optionally, we can define an assessor function. At each generation, after
# fitness is calculated for each organism, each organism is assessed using the
# assessor function we define. If the assessor function returns True, the NEAT
# search will terminate. A list of organism which 'solve' our assessor function
# can be retrieve from the NEAT object via the get_solvers() method.
def score_assessor(organism):
    score = sum([abs(y - organism.think(x)[0]) < 0.5 for x, y in DATA])
    return score == len(DATA)


# Note that we have access to information inside the organism objects, we could,
# for example, only terminate our search once the fitness reaches a certain threshold.
def fitness_assessor(organism):
    return organism.fitness > 15


if __name__ == '__main__':

    # Number of inputs and outputs for the network
    num_inputs = 2
    num_outputs = 1

    # Our subclassed organism type
    organism_type = XOR

    neat = Neat(num_inputs, num_outputs, organism_type, population_size=500, assessor_function=score_assessor)

    # If we don't specify a number of iteration, the search runs indefinitely
    neat.run(iterations=200)

    # Retrieve the solvers
    solvers = neat.get_solvers()

    if solvers:

        # Test out the solver by providing inputs to the think method
        print("Solver outputs for XOR data:")
        print("Fitness:", solvers[0].fitness)
        print("Hidden Nodes:", len(solvers[0].brain.hidden_nodes))
        for x, _ in DATA:
            print(solvers[0].think(x))

        # Now enjoy your trained network!