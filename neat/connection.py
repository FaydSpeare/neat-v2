import random
import itertools

connection_innovation_counter = itertools.count()

class Connection:

    def __init__(self, input_node, output_node, number=None):
        self.input_node = input_node
        self.output_node = output_node
        self.number = next(connection_innovation_counter) if number is None else number
        self.weight = None
        self.enabled = True
        self.init_weight()

    def get_output(self):
        return self.input_node.get_output() * self.weight if self.enabled else 0.

    def disable(self):
        self.enabled = False

    def get_input_node(self):
        return self.input_node

    def get_output_node(self):
        return self.output_node

    def set_weight(self, weight):
        self.weight = weight

    def backward(self, layer):
        self.input_node.backward(layer - 1)

    def init_weight(self):
        self.weight = random.gauss(0, 1)

    def mutate_weight(self):
        self.weight += 0.01 * random.gauss(0, 1)
