# Adapted from https://gist.github.com/betandr/541a1f6466b6855471de5ca30b74cb31
from decimal import Decimal


class Edge:
    def __init__(self, to_node, length):
        self.to_node = to_node
        self.length = length

    def to_dict(self):
        return {'to_node': self.to_node, 'length': self.length}

    @staticmethod
    def from_dict(edge_dict):
        return Edge(edge_dict['to_node'], edge_dict['length'])


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = dict()

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, from_node, to_node, length):
        edge = Edge(to_node, length)
        # edge = to_node
        if from_node in self.edges:
            from_node_edges = self.edges[from_node]
        else:
            self.edges[from_node] = dict()
            from_node_edges = self.edges[from_node]
        # if edge not in from_node_edges:
        # from_node_edges.append(edge)
        from_node_edges[to_node] = edge

    def to_dict(self):
        return {
            'nodes': list(self.nodes),
            'edges': {from_node: {to_node: edge.to_dict() for to_node, edge in edges.items()}
                      for from_node, edges in self.edges.items()}
        }

    @staticmethod
    def from_dict(graph_dict):
        graph = Graph()
        graph.nodes = set(graph_dict['nodes'])
        graph.edges = {from_node: {to_node: Edge.from_dict(edge_dict)
                                   for to_node, edge_dict in edges.items()}
                       for from_node, edges in graph_dict['edges'].items()}
        return graph

def min_dist(q, dist):
    """
    Returns the node with the smallest distance in q.
    Implemented to keep the main algorithm clean.
    """
    min_node = None
    for node in q:
        if min_node == None:
            min_node = node
        elif dist[node] < dist[min_node]:
            min_node = node

    return min_node


INFINITY = float('Infinity')


def dijkstra(graph, source):
    q = set()
    dist = {}
    prev = {}

    for v in graph.nodes:       # initialization
        dist[v] = INFINITY      # unknown distance from source to v
        prev[v] = INFINITY      # previous node in optimal path from source
        q.add(v)                # all nodes initially in q (unvisited nodes)

    # distance from source to source
    dist[source] = 0

    while q:
        # node with the least distance selected first
        u = min_dist(q, dist)

        q.remove(u)

        try:
            if u in graph.edges:
                for _, v in graph.edges[u].items():
                    alt = dist[u] + v.length
                    if alt < dist[v.to_node]:
                        # a shorter path to v has been found
                        dist[v.to_node] = alt
                        prev[v.to_node] = u
        except:
            pass

    return dist, prev


def to_array(prev, from_node):
    """Creates an ordered list of labels as a route."""
    previous_node = prev[from_node]
    route = [from_node]

    while previous_node != INFINITY:
        route.append(previous_node)
        temp = previous_node
        previous_node = prev[temp]

    route.reverse()
    return route
