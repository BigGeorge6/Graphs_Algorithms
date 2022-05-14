import sys
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)
        
    def construct_graph(self, nodes, init_graph):
        '''
        This method makes sure that the graph is symmetrical. In other words, if there's a path from node A to B with a value V, there needs to be a path from node B to node A with a value V.
        '''
        graph = {}
        for node in nodes:
            graph[node] = {}
        
        graph.update(init_graph)
        
        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value
                    
        return graph
    
    def get_nodes(self):
        "Returns the nodes of the graph."
        return self.nodes
    
    def get_outgoing_edges(self, node):
        "Returns the neighbors of a node."
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections
    
    def value(self, node1, node2):
        "Returns the value of an edge between two nodes."
        return self.graph[node1][node2]


def dijkstra_algorithm(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())
 
    # We'll use this dict to save the cost of visiting each node and update it as we move along the graph   
    shortest_path = {}
 
    # We'll use this dict to save the shortest known path to a node found so far
    previous_nodes = {}
 
    # We'll use max_value to initialize the "infinity" value of the unvisited nodes   
    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    # However, we initialize the starting node's value with 0   
    shortest_path[start_node] = 0
    
    # The algorithm executes until we visit all nodes
    while unvisited_nodes:
        # The code block below finds the node with the lowest score
        current_min_node = None
        for node in unvisited_nodes: # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node
                
        # The code block below retrieves the current node's neighbors and updates their distances
        neighbors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node
 
        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)
    
    return previous_nodes, shortest_path


def print_result(previous_nodes, shortest_path, start_node, target_node):
    path = []
    node = target_node
    
    while node != start_node:
        path.append(node)
        node = previous_nodes[node]
 
    # Add the start node manually
    path.append(start_node)
    
    print("We found the following best path with a value of {}.".format(shortest_path[target_node]))
    print(" -> ".join(reversed(path)))

nodes = ["Invitarla", "Pasar por ella", "Esperar que llegue", "Ir al lugar", 
        "Comer", "Beber", "Coquetearle", "Acariciarle","Besarla","Sexo <3"]
 
init_graph = {}
for node in nodes:
    init_graph[node] = {}

distances = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for x in range(15):
    distances[x] = random.randint(0,100)
  
init_graph["Invitarla"]["Ir al lugar"] = distances[0]
init_graph["Ir al lugar"]["Acariciarle"] = distances[1]
init_graph["Ir al lugar"]["Coquetearle"] = distances[2]
init_graph["Acariciarle"]["Besarla"] = distances[3]
init_graph["Besarla"]["Coquetearle"] = distances[4]
init_graph["Coquetearle"]["Sexo <3"] = distances[5]
init_graph["Ir al lugar"]["Beber"] = distances[6]
init_graph["Beber"]["Sexo <3"] = distances[7]
init_graph["Invitarla"]["Esperar que llegue"] = distances[8]
init_graph["Invitarla"]["Pasar por ella"] = distances[9]
init_graph["Pasar por ella"]["Esperar que llegue"] = distances[10]
init_graph["Esperar que llegue"]["Beber"] = distances[11]
init_graph["Esperar que llegue"]["Comer"] = distances[12]
init_graph["Comer"]["Beber"] = distances[13]

graph = Graph(nodes, init_graph)
previous_nodes, shortest_path = dijkstra_algorithm(graph=graph, start_node="Invitarla")
print_result(previous_nodes, shortest_path, start_node="Invitarla", target_node="Sexo <3")

G = nx.Graph()

'''
1.Invitarla 2.Pasar por ella 3.Esperar que llegue
4.Ir al lugar 5.Comer 6.Beber 7.Coquetearle
8.Acariciarle 9.Besarla 10.Sexo <3
'''  
G.add_node("Invitarla",pos=(0,6))
G.add_node("Pasar por ella",pos=(8,0))
G.add_node("Esperar que llegue",pos=(11,4))
G.add_node("Ir al lugar",pos=(6,11))
G.add_node("Comer",pos=(15,0))
G.add_node("Beber",pos=(16,6))
G.add_node("Coquetearle",pos=(17,10))
G.add_node("Acariciarle",pos=(10,15))
G.add_node("Besarla",pos=(16,13))
G.add_node("Sexo <3",pos=(20,8))

G.add_edge("Invitarla","Ir al lugar", energy = distances[0])
G.add_edge("Ir al lugar","Acariciarle", energy = distances[1])
G.add_edge("Ir al lugar","Coquetearle", energy = distances[2])
G.add_edge("Acariciarle","Besarla", energy = distances[3])
G.add_edge("Besarla","Coquetearle", energy = distances[4])
G.add_edge("Coquetearle","Sexo <3", energy = distances[5])
G.add_edge("Ir al lugar","Beber", energy = distances[6])
G.add_edge("Beber","Sexo <3", energy = distances[7])
G.add_edge("Invitarla","Esperar que llegue", energy = distances[8])
G.add_edge("Invitarla","Pasar por ella", energy = distances[9])
G.add_edge("Pasar por ella","Esperar que llegue", energy = distances[10])
G.add_edge("Esperar que llegue","Beber",energy = distances[11])
G.add_edge("Esperar que llegue","Comer",energy = distances[12])
G.add_edge("Comer","Beber", energy = distances[13])

pos = nx.get_node_attributes(G,'pos')
labels = nx.get_edge_attributes(G,'energy')

nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
nx.draw(G, pos, with_labels=True)

print(nx.shortest_path(G, source="Invitarla",target="Sexo <3"))
plt.show()