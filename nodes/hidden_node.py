import math

from nodes.node import Node

class Hidden(Node):

    def __init__(self, number=None):
        super().__init__(number) # TODO what is the layer?

    def replicate(self):
        return Hidden(self.number)

