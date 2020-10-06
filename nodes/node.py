import itertools
import math

node_innovation_counter = itertools.count(1)

class Node:

    def __init__(self, number):
        self.layer = 0
        self.number = next(node_innovation_counter) if number is None else number
        self.sum = 0.
        self.output = 0.
        self.connections = list()
        self.activated = False


    def get_output(self):
        if self.activated:
            return self.output
        for connection in self.connections:
            self.sum += connection.get_output()
        self.activated = True
        exponent = -4.9 * self.sum
        exponent = max(-100., exponent)
        exponent = min(100., exponent)
        self.output = 1 / (1 + math.exp(exponent))
        return self.output


    def reset(self):
        self.layer = 0
        self.activated = False
        self.output = 0.


    def add_connection(self, connection):
        self.connections.append(connection)


    def backward(self, layer):
        self.layer = min(self.layer, layer)
        for connection in self.connections:
            connection.backward(layer)


    def get_layer(self):
        return self.layer


    def set_layer(self, layer):
        self.layer = layer


    def is_connected_to(self, node):
        return node in [connection.input_node for connection in self.connections]



