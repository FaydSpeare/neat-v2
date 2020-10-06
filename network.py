import random

from nodes.bias_node import Bias
from nodes.input_node import Input
from nodes.output_node import Output
from nodes.hidden_node import Hidden
from connection import Connection

class Network:

    def __init__(self, num_inputs, num_outputs, replication=False):

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.bias = Bias()
        self.input_nodes = list()
        self.output_nodes = list()
        self.hidden_nodes = list()

        self.bias_connections = list()
        self.connections = list()
        self.splitable_connections = set()

        if not replication: self.create_initial_connections()


    def create_initial_connections(self):
        self.input_nodes = [Input() for _ in range(self.num_inputs)]
        self.output_nodes = [Output() for _ in range(self.num_outputs)]
        for out_node in self.output_nodes:
            bias_connection = Connection(self.bias, out_node)
            self.bias_connections.append(bias_connection)
            out_node.add_connection(bias_connection)
            for in_node in self.input_nodes:
                connection = Connection(in_node, out_node)
                self.connections.append(connection)
                out_node.add_connection(connection)
                self.splitable_connections.add(connection)


    def reset(self):
        [node.reset() for node in self.input_nodes]
        [node.reset() for node in self.hidden_nodes]
        [node.reset() for node in self.output_nodes]


    def forward(self, inputs):

        # Reset all nodes
        self.reset()

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
        self.hidden_nodes.append(new_node)

        first_connection = Connection(input_node, new_node)
        self.connections.append(first_connection)
        self.splitable_connections.add(first_connection)
        first_connection.set_weight(1)

        second_connection = Connection(new_node, output_node)
        self.connections.append(second_connection)
        self.splitable_connections.add(second_connection)

        bias_connection = Connection(self.bias, new_node)
        self.bias_connections.append(bias_connection)

        # Add new connections to nodes
        new_node.add_connection(first_connection)
        new_node.add_connection(bias_connection)
        output_node.add_connection(second_connection)


    def add_connection(self):

        input_layer = self.calibrate_layers()
        all_nodes = self.input_nodes + self.hidden_nodes + self.output_nodes

        for i in range(10):

            in_node, out_node = random.sample(all_nodes, 2)
            if out_node.get_layer() < in_node.get_layer(): in_node, out_node = out_node, in_node

            # Can't connect to an input node
            if out_node.get_layer() != input_layer:

                # Can't connect from output node
                if in_node.get_layer() != 0:

                    # Can't connect if connection already exists
                    if not out_node.is_connected_to(in_node):

                        new_connection = Connection(in_node, out_node)
                        self.connections.append(new_connection)
                        self.splitable_connections.add(new_connection)
                        out_node.add_connection(new_connection)
                        break


    def calibrate_layers(self):

        # Reset all nodes
        self.reset()

        for out_node in self.output_nodes:
            out_node.backward(0)

        # Set true input layer
        true_input_layer = min([node.get_layer() for node in self.input_nodes])
        [node.set_layer(true_input_layer) for node in self.input_nodes]
        return true_input_layer


    def replicate(self):

        copy = Network(self.num_inputs, self.num_outputs, replication=True)

        innovation_number_to_node = {0 : copy.bias}

        # Copy Input Nodes
        for node in self.input_nodes:
            node_copy = node.replicate()
            copy.input_nodes.append(node_copy)
            innovation_number_to_node[node_copy.number] = node_copy

        # Copy Hidden Nodes
        for node in self.hidden_nodes:
            node_copy = node.replicate()
            copy.hidden_nodes.append(node_copy)
            innovation_number_to_node[node_copy.number] = node_copy

        # Copy Output Nodes
        for node in self.output_nodes:
            node_copy = node.replicate()
            copy.output_nodes.append(node_copy)
            innovation_number_to_node[node_copy.number] = node_copy

        # Copy Connections
        for out_node in self.hidden_nodes + self.output_nodes:
            node_copy = innovation_number_to_node[out_node.number]

            for connection in out_node.connections:

                input_node_copy = innovation_number_to_node[connection.input_node.number]
                output_node_copy = innovation_number_to_node[connection.output_node.number]

                connection_copy = Connection(input_node_copy, output_node_copy, number=connection.number)
                connection_copy.enabled = connection.enabled
                connection_copy.weight = connection.weight

                node_copy.add_connection(connection_copy)

                if connection in self.splitable_connections:
                    copy.splitable_connections.add(connection_copy)

                if input_node_copy.number == 0:
                    copy.bias_connections.append(connection_copy)
                else:
                    copy.connections.append(connection_copy)

        return copy


    def full_weight_mutation(self):
        for connection in self.connections + self.bias_connections:
            connection.init_weight()


    def mutate(self):
        if random.random() < 0.9:
            for c in self.connections + self.bias_connections:
                if random.random() < 0.9: c.mutate_weight()
                else: c.init_weight()

        if random.random() < 0.05:
            self.add_connection()

        if random.random() < 0.01:
            self.add_node()



if __name__ == '__main__':
    net = Network(3, 80)

    for i in range(20):
        net.add_node()
        net.add_connection()

    copy = net.replicate()
    print(net.forward([1, 1, 1]))
    print(copy.forward([1, 1, 1]))





