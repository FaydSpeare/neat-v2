import random

from neat.nodes.bias_node import Bias
from neat.nodes.input_node import Input
from neat.nodes.output_node import Output
from neat.nodes.hidden_node import Hidden
from neat.connection import Connection
from neat.structure_tracker import register_node, register_connection, get_connection_innovation_number, get_node_innovation_number

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


    def create_node(self, connection):

        # Remove connection so it's not split again
        # TODO Can we split connections multiple times? -> This could break the structure_tracker as splitting the connection
        # TODO again would give the same node innovation number. (this may only be the case if structure mutations are
        # TODO tracked across multiple generations).
        self.splitable_connections.remove(connection)

        # Disable existing connection
        connection.disable()

        # Reuse node innovation number from previous identical mutation
        number = get_node_innovation_number(connection.number)

        # Create new Hidden node
        new_node = Hidden(number=number)
        self.hidden_nodes.append(new_node)

        # Save number for future potential duplicate structural mutations
        if number is None: register_node(connection.number, new_node.number)

        return new_node


    def create_connection(self, inode, onode, splittable=True):

        # Reuse node innovation number from previous identical mutation if possible
        number = get_connection_innovation_number(inode.number, onode.number)

        connection = Connection(inode, onode, number=number)
        self.connections.append(connection)

        # Connect to out node
        onode.add_connection(connection)

        if splittable: self.splitable_connections.add(connection)

        # Save number for future potential duplicate structural mutations
        if number is None: register_connection(inode.number, onode.number, connection.number)

        return connection


    def add_node(self):

        # Ignore if there are not connections to be split
        if len(self.splitable_connections) == 0: return

        # Randomly sample one connection to split
        splitable_connection = random.sample(self.splitable_connections, 1)[0]
        input_node = splitable_connection.get_input_node()
        output_node = splitable_connection.get_output_node()

        # Create new node
        new_node = self.create_node(splitable_connection)

        # Create new connections
        first_connection = self.create_connection(input_node, new_node, splittable=True)
        second_connection = self.create_connection(new_node, output_node, splittable=True)
        bias_connection = self.create_connection(self.bias, new_node, splittable=False)

        # Set weight of first connection to 1
        first_connection.set_weight(1.)


    def add_connection(self):

        input_layer = self.calibrate_layers()
        all_nodes = self.input_nodes + self.hidden_nodes + self.output_nodes

        # Retry this many times.
        for _ in range(10):

            # Randomly select 2 nodes
            in_node, out_node = random.sample(all_nodes, 2)

            # Reverse the order of nodes if they aren't feeding forward
            if out_node.get_layer() < in_node.get_layer():
                in_node, out_node = out_node, in_node

            # Can't connect to an input node
            if out_node.get_layer() != input_layer:

                # Can't connect from output node
                if in_node.get_layer() != 0:

                    # Can't connect if connection already exists
                    if not out_node.is_connected_to(in_node):

                        new_connection = self.create_connection(in_node, out_node, splittable=True)
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

        if random.random() < 0.8:
            for c in self.connections + self.bias_connections:
                if random.random() < 0.9:
                    c.mutate_weight()
                else:
                    c.init_weight()

        if random.random() < 0.1:
            self.add_connection()

        if random.random() < 0.01:
            self.add_node()


    def get_connections(self):
        return self.connections + self.bias_connections



if __name__ == '__main__':
    net = Network(3, 80)

    for i in range(20):
        net.add_node()
        net.add_connection()

    copy = net.replicate()
    print(net.forward([1, 1, 1]))
    print(copy.forward([1, 1, 1]))





