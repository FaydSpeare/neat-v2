import itertools

node_innovation_counter = itertools.count(1)

class Node:

    def __init__(self, activation_func, number):
        self.activation_func = activation_func
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
        self.activate()
        return self.output

    def activate(self):
        self.activated = True
        self.output = self.activation_func(self.sum)

    def reset(self):
        self.layer = 0
        self.activated = False
        self.output = 0.
        self.sum = 0


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




