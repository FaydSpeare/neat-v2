from neat.nodes.node import Node

class Hidden(Node):

    def __init__(self, activation_func, number=None):
        super().__init__(activation_func, number) # TODO what is the layer?

    def replicate(self):
        return Hidden(self.activation_func, self.number)

