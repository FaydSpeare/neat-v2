import random
import itertools

connection_innovation_counter = itertools.count()

class Connection:

    def __init__(self, input_node, output_node, config, number=None):
        self.input_node = input_node
        self.output_node = output_node
        self.config = config
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

    def get_perturbation(self):
        if self.config['weight_init'] == 'gauss':
            return random.gauss(*self.config['weight_init_params'])
        if self.config['weight_init'] == 'uniform':
            return random.uniform(*self.config['weight_init_params'])

    def init_weight(self):
        self.weight = self.get_perturbation()

    def mutate_weight(self):
        self.weight += self.config['weight_mutate_step'] * self.get_perturbation()
