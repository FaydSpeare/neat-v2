from network import Network

class Organism:

    def __init__(self, num_inputs, num_outputs, replication=False):

        self.brain = None
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        if not replication:
            self.brain = Network(num_inputs, num_inputs)

    def think(self, sensors):
        return self.brain.forward(sensors)

    def replicate(self):
        copy = self.__class__(self.num_inputs, self.num_outputs, replication=True)
        copy.brain = self.brain.replicate()

