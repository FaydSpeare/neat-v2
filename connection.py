import random
import itertools

connection_innovation_counter = itertools.count()

class Connection:

    def __init__(self, input_node, output_node):
        self.input_node = input_node
        self.output_node = output_node
        self.innovation_number = next(connection_innovation_counter)
        self.weight = random.gauss(0, 1)
        self.enabled = True

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