VERBOSE = False

connection_innovations = dict()

def register_connection(in_node, out_node, connection_innovation):
    global connection_innovations
    connection_innovations[(in_node, out_node)] = connection_innovation

def get_connection_innovation_number(in_node, out_node):
    global connection_innovations
    if VERBOSE and (in_node, out_node) in connection_innovations:
        print("saved CONNECTION number", connection_innovations[(in_node, out_node)])
    return connection_innovations.get((in_node, out_node))


node_innovations = dict()

def register_node(connection, node_innovation):
    global node_innovations
    node_innovations[connection] = node_innovation

def get_node_innovation_number(connection):
    global node_innovations
    if VERBOSE and connection in node_innovations:
        print("saved NODE number", node_innovations[connection])
    return node_innovations.get(connection)


def reset_structure_tracker():
    global node_innovations
    global connection_innovations
    node_innovations = dict()
    connection_innovations = dict()