from neat.nodes.node import Node

class Bias(Node):

    def __init__(self):
        super().__init__(None, 0)

    def replicate(self):
        return Bias()

    def get_output(self):
        return 1.