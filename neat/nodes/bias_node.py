from neat.nodes.node import Node

class Bias(Node):

    def __init__(self):
        super().__init__(None, 0)

    def replicate(self):
        return Bias()

    def set_input(self, output):
        self.activated = True
        self.output = output

    def get_output(self):
        if not self.activated:
            raise Exception("Input Node has not been activated")
        return self.output