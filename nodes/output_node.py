import math

from nodes.node import Node

class Output(Node):

    def __init__(self, number=None):
        super().__init__(number)

    def replicate(self):
        return Output(self.number)