import copy, pickle, os, config, pdb
import numpy as np
from subgraph import Subgraph
import random

class Graph():

    def __init__(self, size, nodes, edges, name = 'enron'):

        self.size = size
        self.nodeThetas = {}    # Key = node id, value = theta.
        self.edges = [ [] for i in range(size)]    # indexed by id, list of neighbor ids.
        self.distM = None # size x size matrix of distances. Distance metric counting each edge as dist 1.
        self.distMP = None # distance metric with regard to 1/ edge prob.
        self.distMP2 = None # Distance metric when using constant prob on all edges
        
        self.name = name
        self.edgeProb = {} # (i,j) edge indexes the edge probability.
        self.all_paths = None

        # Rename the nodes so ids go from 0 to size.
        count = 0
        idToNum = {}
        for nodeId, theta in nodes:
            idToNum[nodeId] = count
            self.nodeThetas[count] = theta
            count += 1

        for i, j, p in edges:
            self.edges[idToNum[i]].append(idToNum[j])
            self.edgeProb[(idToNum[i], idToNum[j])] = p

            self.edges[idToNum[j]].append(idToNum[i])
            self.edgeProb[(idToNum[j], idToNum[i])] = p


    # Returns clusters in the graph based on the theta value input.
    def __get_clusters(self, theta):

        clusters = []

        visited = set()
        stack = []

        for nodeId in range(self.size):

            currCluster = set()

            if nodeId not in visited and self.nodeThetas[nodeId] <= theta:
                # This node is a good node.
                stack.append(nodeId)
                visited.add(nodeId)

            while len(stack) > 0:
                currNode = stack.pop()
                currCluster.add(currNode)

                for neigh in self.edges[currNode]:
                    currCluster.add(neigh)
                    if neigh not in visited and self.nodeThetas[neigh] <= theta:
                        stack.append(neigh)
                        visited.add(neigh)

            
            if len(currCluster) > 0:    
                clusters.append(currCluster)

        return clusters


    # Returns a new graph that has only the largest cluster elements.
    def get_largest_cluster_subgraph(self, theta):

        clusters = self.__get_clusters(theta)

        size = 0
        largest = None
        for cluster in clusters:
            if len(cluster) > size:
                largest = cluster
                size = len(cluster)

        size = len(largest)
        nodes = []
        for nId in largest:
            nodes.append((nId, self.nodeThetas[nId]))

        edges = []

        for i in largest:
            for j in self.edges[i]:
                if j in largest:
                    edges.append( (i, j, self.edgeProb[(i,j)]))

            
        name = self.name + "_{0}".format(theta)

        subgraph = Graph(size, nodes, edges, name)

        return subgraph


    # Returns the inner and outer nodes of graph.
    def get_inner_outer_nodes(self, theta):
        inner = []
        outer = []

        for nId in self.nodeThetas:
            if self.nodeThetas[nId] <= theta:
                inner.append(nId)
            else:
                outer.append(nId)

        return inner, outer

    
    # Runs Floyd-Warshall alg. to calculate all-pairs shortest paths.
    def get_distance_matrix(self):

        dist_matrix_file = config.dist_matrix.format(self.name)

        if self.distM:
            return self.distM

        if os.path.isfile(dist_matrix_file):
            self.distM = pickle.load(open(dist_matrix_file, 'rb'))
            return self.distM

        else:
            self.__calculate_distance_matrix()
            pickle.dump(self.distM, open(dist_matrix_file, 'wb'))
            return self.distM

    def __calculate_distance_matrix(self):

        size = self.size
        #self.distM = np.zeros(shape= (size, size))
        self.distM = [[0]* size for i in range(size)]
        self.distMP = [[0]* size for i in range(size)]

        for i in range(size):
            for j in range(size):
                if j in self.edges[i]:
                    self.distMP[i][j] = 1.0 / self.edgeProb[(i,j)]
                else:
                    self.distMP[i][j] = 1 / 0.01    # Approximation for what "infinity is"
                if j in self.edges[i]:
                    self.distM[i][j] = 1.0
                else:
                    self.distM[i][j] = size +1   # Approximation for what "infinity is"

        for k in range(size):
            for i in range(size):
                for j in range(size):
                    if (self.distM[i][k] + self.distM[k][j] < self.distM[i][j]):
                        self.distM[i][j] = self.distM[i][k] + self.distM[k][j]

                    if (self.distMP[i][k] + self.distMP[k][j] < self.distMP[i][j]):
                        self.distMP[i][j] = self.distMP[i][k] + self.distMP[k][j]

        for i in range(size):
            for j in range(size):
                if self.distM[i][j] == size + 1:
                    self.distM[i][j] = None

    def calculate_distance_matrix_with_prob(self, p):
        self.distMP2 = [[0]* self.size for i in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                if self.distM[i][j] == None:
                    self.distMP2[i][j] = None
                else:
                    self.distMP2[i][j] = 1.0 /(p ** self.distM[i][j])
        
            
    # Function to reset edge probabilities based on input distribution function.
    def set_edge_probabilities(self, func):
        for (i,j) in self.edgeProb:
            self.edgeProb[(i,j)] = func(i, j)
                        
    
    # Returns for a given theta, the deterministic subgraph when having chosen
    # edges according to probability.
    def get_subgraph(self, theta):
        inner, outer = self.get_inner_outer_nodes(theta)
        edges = [[] * self.size for i in range(self.size)]

        for i in range(self.size):
            for j in self.edges[i]:
                if i < j:
                    rand = random.random()
                    if rand >= self.edgeProb[(i,j)]:
                        edges[i].append(j)
                        edges[j].append(i)

        return Subgraph(edges, inner, outer)

    def __get_all_paths(self, node, paths, visited, depth):
        if node in visited:
            return

        if node in paths:
            paths[node].append(depth)
        else:
            paths[node] = [depth]

        visited.add(node)
        for j in self.edges[node]:
            self.__get_all_paths(j, paths, visited, depth+1)
        visited.remove(node)
        return
        

    def calculate_all_paths(self):
        paths = []

        for i in range(self.size):
            print i
            node_paths = {}
            visited = set()
            visited.add(i)
            for j in self.edges[i]:
                self.__get_all_paths(j, node_paths, visited, 1)

            paths.append(node_paths)

        self.all_paths = paths



if __name__ == '__main__':
    edges = [(0,2, 1),
             (1,2,1),
             (1,3,1),
             (2,3,1),
             (2,4,1),
             (3,4,1),
             (4,5,1),]

    nodes = [(0,1),
             (1,1),
             (2,1),
             (3,1),
             (4,1),
             (5,1)]

    size = 6

    G = Graph(size, nodes, edges)

    G.calculate_all_paths()

    print G.all_paths
        

    
