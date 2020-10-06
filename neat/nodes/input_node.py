from neat.nodes.node import Node

class Input(Node):

    def __init__(self, number=None):
        super().__init__(number)

    def replicate(self):
        return Input(self.number)

    def set_input(self, output):
        self.activated = True
        self.output = output

    def get_output(self):
        if not self.activated:
            raise Exception("Input Node has not been activated")
        return self.output