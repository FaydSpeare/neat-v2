from nodes.node import Node

class Bias(Node):

    def __init__(self, number=None):
        super().__init__(number)

    def replicate(self):
        return Bias(self.number)

    def get_output(self):
        return 1.