import random

from nodes.bias_node import Bias
from nodes.input_node import Input
from nodes.output_node import Output
from nodes.hidden_node import Hidden
from connection import Connection

class Network:

    def __init__(self, num_inputs, num_outputs):

        self.bias = Bias()
        self.input_nodes = [Input() for _ in range(num_inputs)]
        self.output_nodes = [Output() for _ in range(num_outputs)]

        self.nodes_list = list()
        self.nodes_list.extend(self.input_nodes)
        self.nodes_list.extend(self.output_nodes)

        self.splitable_connections = set()

        for out_node in self.output_nodes:
            out_node.add_connection(Connection(self.bias, out_node))
            for in_node in self.input_nodes:
                connection = Connection(in_node, out_node)
                out_node.add_connection(connection)
                self.splitable_connections.add(connection)


    def forward(self, inputs):

        # Reset all nodes
        [node.reset() for node in self.nodes_list]

        # Set Inputs
        [node.set_input(inputs[i]) for i, node in enumerate(self.input_nodes)]

        # Get Outputs
        return [node.get_output() for node in self.output_nodes]


    def add_node(self):

        if len(self.splitable_connections) == 0:
            return

        splitable_connection = random.sample(self.splitable_connections, 1)[0]
        input_node = splitable_connection.get_input_node()
        output_node = splitable_connection.get_output_node()

        # Disable existing connection
        splitable_connection.disable()

        # Create new Hidden node
        new_node = Hidden()
        self.nodes_list.append(new_node)

        first_connection = Connection(input_node, new_node)
        self.splitable_connections.add(first_connection)
        first_connection.set_weight(1)

        second_connection = Connection(new_node, output_node)
        self.splitable_connections.add(second_connection)

        bias_connection = Connection(self.bias, new_node)

        # Add new connections to nodes
        new_node.add_connection(first_connection)
        new_node.add_connection(bias_connection)
        output_node.add_connection(second_connection)


    def add_connection(self):

        input_layer = self.calibrate_layers()

        for i in range(10):

            in_node, out_node = random.sample(self.nodes_list, 2)
            if out_node.get_layer() < in_node.get_layer(): in_node, out_node = out_node, in_node

            # Can't connect to an input node
            if out_node.get_layer() != input_layer:

                # Can't connect if connection already exists
                if not out_node.is_connected_to(in_node):

                    print(in_node.number, in_node.layer, "->", out_node.number, out_node.layer)
                    new_connection = Connection(in_node, out_node)
                    self.splitable_connections.add(new_connection)
                    out_node.add_connection(new_connection)
                    break


    def calibrate_layers(self):

        # Reset all nodes
        [node.reset() for node in self.nodes_list]

        for out_node in self.output_nodes:
            out_node.backward(0)

        # Set true input layer
        true_input_layer = min([node.get_layer() for node in self.input_nodes])
        [node.set_layer(true_input_layer) for node in self.input_nodes]
        return true_input_layer







if __name__ == '__main__':
    net = Network(3, 1)
    net.add_node()
    net.calibrate_layers()
    net.add_connection()
    net.calibrate_layers()
    print(net.forward([0, 0, 0]))





