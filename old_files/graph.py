import copy, pickle, os, config, pdb
import numpy as np
from subgraph import Subgraph

class Graph():

    def __init__(self, edges, num_nodes, name, node_params = None):
        self.size = num_nodes
        self.node_set = set()
        self.edge_size = len(edges)
        self.distM = None
        self.node_params = {}
        self.name = name


        self.M = {}
        
        for (i, j, w) in edges:
            if i not in self.M:
                self.M[i] = {}

            self.M[i][j] = w

            if j not in self.M:
                self.M[j] = {}

            self.M[j][i] = w

            self.node_set.add(i)
            self.node_set.add(j)

        self.nodes = [i for i in self.node_set]
        self.nodes.sort()

        self.__get_distance_matrix()

        if node_params:
            self.node_params = node_params
        else:
            self.__set_node_params()


    # Use Floyd-Warshall alg to find all-pairs distance.
    def __get_distance_matrix(self):

        dist_matrix_file = config.dist_matrix.format(self.name)

        if os.path.isfile(dist_matrix_file):
            self.distM = pickle.load(open(dist_matrix_file, 'rb'))
            return
        
        self.distM = copy.deepcopy(self.M)

        for k in (self.nodes):
            for i in (self.nodes):
                for j in (self.nodes):

                    if j not in self.distM[i] and (k not in self.distM[i] or
                                                   j not in self.distM[k]):
                        continue

                    elif j not in self.distM[i]:
                        self.distM[i][j] = self.distM[i][k] + self.distM[k][j]

                    elif (k not in self.distM[i] or
                          j not in self.distM[k]):
                        
                        self.distM[i][j] = self.distM[i][j]
                    else:
                        self.distM[i][j] = min(self.distM[i][j],
                                               self.distM[i][k] + self.distM[k][j])

        pickle.dump(self.distM, open(dist_matrix_file, 'wb'))
                        

    # Return distance between two nodes.
    def get_dist(self, i, j):

        if not self.distM:
            self.__get_distance_matrix()

        if j in self.distM[i]:
            return self.distM[i][j]
        else:
            return None


    # Sets the node parameters such as critical value.
    def __set_node_params(self, distribution= 0):

        if distribution == 0:
            # Uniform distribution
            s = np.random.uniform(0, 1, len(self.nodes))

        elif distribution == 1:
            # Normal distribution
            s = np.random.normal(0.5, 0.4, len(self.nodes))
            s = np.clip(s, 0, 1)
        
        for i, k in enumerate(self.nodes):
            self.node_params[k] = s[i]


    # Finds the cluster which this node belongs to.
    def __get_cluster(self, n, theta):
        visited = set()
        queue = [n]
        visited.add(n)
        inner = []
        outer = []

        while len(queue) > 0:
            i = queue.pop()
            if self.node_params[i] > theta:
                outer.append(i)
            else:
                inner.append(i)
                for j in self.M[i]:
                    if j not in visited:
                        visited.add(j)
                        queue.append(j)

        return [inner, outer]   

    
    def get_clusters(self, theta):
        cluster_dict = {}
        clusters = []

        # Boundary nodes
        for i in self.nodes:
            cluster_dict[i] = []


                
        for i in self.nodes:
            if len(cluster_dict[i]) > 0 or self.node_params[i] > theta:
                continue

            cluster = self.__get_cluster(i, theta)
            clusters.append(cluster)

            for n in cluster[0]:
                cluster_dict[n].append(len(clusters) - 1)

            for n in cluster[1]:
                cluster_dict[n].append(len(clusters) - 1)

        return clusters, cluster_dict


    def get_largest_cluster(self, theta):
        clusters, cluster_dict = self.get_clusters(theta)
        largest = None
        size = 0

        for cluster in clusters:
            if len(cluster[0]) + len(cluster[1]) > size:
                largest = cluster
                size =  len(cluster[0]) + len(cluster[1])

        return largest


    def check_clusters(self, clusters, cluster_dict):

        inner_set = set()
        outer_set = set()
        
        for inner, outer in clusters:
            for n in inner:
                if n in inner_set or n in outer_set:
                    pdb.set_trace()

                inner_set.add(n)

            for n in outer:
                if n in inner_set:
                    pdb.set_trace()

                outer_set.add(n)


    def get_subgraph(self, cluster, distribution, name):
        inner = set(cluster[0])
        outer = set(cluster[1])

        nodes = inner.union(outer)

        edges = []
        for i in nodes:
            for j in self.M[i]:

                if i > j:
                    continue
                if j in nodes:
                    edges.append((i, j, self.M[i][j]))

        edge_size = len(edges)

        prob_edges = []

        randA = np.random.uniform(0, 1, edge_size)

        if distribution <= 1:
            s = randA <= distribution
            
        if distribution == 2:
            s = np.clip(np.random.normal(0, 0.4, edge_size), 0, 1)
            s = s >= randA

        for ind, edge in enumerate(edges):
            if s[ind]:
                prob_edges.append(edge)

        node_params = {}

        for n in nodes:
            node_params[n] = self.node_params[n]


        #print "Edges: ", edge_size, len(prob_edges)

        return Subgraph(prob_edges, len(nodes), inner, outer)

        
            
        
                

        

        

            
            

            

            
    
            
        
            
