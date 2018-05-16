import copy, pickle, os, config, pdb
import numpy as np
from subgraph import Subgraph
import random
import networkx as nx

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
        self.btwness = None
        self.katz_vars = None # Used for Katz

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

            #self.edges[idToNum[j]].append(idToNum[i])
            #self.edgeProb[(idToNum[j], idToNum[i])] = p


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
            if self.nodeThetas[i] <= theta:
                for j in self.edges[i]:
                    if j in largest:
                        edges.append( (i, j, self.edgeProb[(i,j)]))
            else:
                # Outer node skip edges
                continue

            
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
                    if rand <= self.edgeProb[(i,j)]:
                        edges[i].append(j)
                        if j not in outer:
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

    def calculate_btwness(self, theta):
        try:
            if self.btwness:
                return
        except Exception as e:
            self.btwness = None
        
        inner, outer = self.get_inner_outer_nodes(theta)
        
        nodes = [ i for i in inner]
        edges = []

        for i in inner:
            for j in self.edges[i]:
                if j in inner:
                    edges.append((i,j))

        G = nx.Graph()

        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        self.btwness = nx.betweenness_centrality(G)


    def _get_A_k(self, A_1, A_prev_k):
        num_inner = len(A_1)
        num_total = len(A_1[0])
        A_k = [[[] for _ in range(num_total)] for _ in range(num_inner)]

        for i in range(num_inner):
            for j in range(num_total):

                if i == j:
                    A_k[i][j] = []
                    continue

                if A_prev_k[i][j] != [] and j < num_inner:
                    for l in range(num_total):
                        if A_1[j][l] == 1 and i != l and l not in A_prev_k[i][j]:
                            A_k[i][l] = A_prev_k[i][j] + [l]# path from i to j
                    
        return A_k

    def calculate_katz(self, theta):
        try:
            if self.katz_vars:
                return
        except Exception as e:
            self.katz_vars = None

        G = self
        
        inner, outer = self.get_inner_outer_nodes(theta)
        num_inner = len(inner)
        num_total = len(inner) + len(outer)
        iterations = 5
        A_mats = [None for _ in range(iterations)]    # Store A_k's
        inner_node_idx_to_id = {node_id: idx for idx, node_id in enumerate(inner)}
        for idx, node_id in enumerate(outer):
            inner_node_idx_to_id[node_id] = idx + len(inner)
            
        # Construct adjacent matrix
        adj_mat = np.zeros((num_inner, num_total))
        for node_idx, inner_node in enumerate(inner):
            for neighbor in G.edges[inner_node]:
                adj_mat[node_idx][inner_node_idx_to_id[neighbor]] = 1
            
        A_mats[0] = [[[j] if adj_mat[i, j] > 0 else [] for j in range(num_total) ]
                     for i in range(num_inner)]

        for i in range(1, iterations):
            A_mats[i] = self._get_A_k(A_mats[0], A_mats[i - 1])
        
        A_binary = []
        for A_i in A_mats:
            cur_A = np.zeros((num_inner, num_total))
            for i in range(num_inner):
                for j in range(num_total):
                    if A_i[i][j] == None:
                        pass
                    else:
                        if len(A_i[i][j]) > 0:
                            if j < num_inner:
                                cur_A[i, j] = 1
                            else:
                                cur_A[i,j] = -1
            A_binary.append(cur_A)


        self.katz_vars = (A_binary, inner, outer, iterations)



if __name__ == '__main__':
    edges = [(0,2),
             (1,2),
             (1,3),
             (2,3),
             (2,4),
             (3,4),
             (4,5),]

    nodes = [0,1,2,3,4,5]
    

    G = nx.Graph()


    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    btw = nx.betweenness_centrality(G)

    print btw
    pdb.set_trace()

    
        

    
