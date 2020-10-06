

connection_innovations = dict()

def register_connection(in_node, out_node, connection_innovation):
    global connection_innovations
    connection_innovations[(in_node, out_node)] = connection_innovation

def get_connection_innovation_number(in_node, out_node):
    global connection_innovations
    return connection_innovations.get((in_node, out_node))


node_innovations = dict()

def register_node(connection, node_innovation):
    global node_innovations
    node_innovations[connection] = node_innovation

def get_node_innovation_number(connection):
    global node_innovations
    return node_innovations.get(connection)