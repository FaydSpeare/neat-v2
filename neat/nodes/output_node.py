from neat.nodes.node import Node

class Output(Node):

    def __init__(self, activation_func, number=None):
        super().__init__(activation_func, number)

    def replicate(self):
        return Output(self.activation_func, self.number)