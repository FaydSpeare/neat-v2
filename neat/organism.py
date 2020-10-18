import random

from neat.network import Network

class Organism:

    def __init__(self, config, replication=False):

        self.brain = None
        self.config = config
        self.num_inputs = config['num_inputs']
        self.num_outputs = config['num_outputs']

        self.fitness = None
        self.adjusted_fitness = None

        if not replication:
            self.brain = Network(self.config)


    def think(self, sensors):
        self.brain.reset()
        return self.brain.forward(sensors)


    def replicate(self):
        copy = self.__class__(self.config, replication=True)
        copy.brain = self.brain.replicate()
        copy.fitness = self.fitness
        return copy


    def fully_weight_mutated(self):
        self.brain.full_weight_mutation()
        return self


    def mutate(self):
        self.brain.mutate()


    def reset(self):
        self.fitness = None
        self.adjusted_fitness = None


    def crossover(self, other_parent):

        # Create copy of this organism (parent with higher fitness)
        offspring = self.replicate()

        for conn in offspring.brain.get_connections():
            for other_conn in other_parent.brain.get_connections():

                # If connection has matching innovation number
                if conn.number == other_conn.number:

                    # Randomly decide which parent's weight to use
                    if random.random() < 0.5:
                        conn.weight = other_conn.weight

                    # disable connection with prob 0.75 if either parent connection is disabled
                    if not conn.enabled or not other_conn.enabled:
                        conn.enabled = not (random.random() < 0.75)

                # disable connection with prob 0.75 parent connection is disabled
                elif not conn.enabled:
                        conn.enabled = not (random.random() < 0.75)

        return offspring


    def __lt__(self, other):
        return self.fitness > other.fitness

