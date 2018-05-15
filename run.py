
import graph, config, pdb
import random, csv, pickle
import numpy as np
import os
import time, heapq

# Parses enron data file.
def get_enron_data():

    edges = []
    nodes = []
    
    with open(config.enron_email_data, 'rb') as f:
        lines = f.readlines()

        for line in lines[4:]:
            lineA = line.split()
            i = int(lineA[0])
            j = int(lineA[1])

            theta1 = get_node_theta(i)
            theta2 = get_node_theta(j)
            p = get_edge_prob(i, j)
            
            edges.append((i, j, p))
            nodes.append((i, theta1))
            nodes.append((j , theta2))

    return len(nodes), nodes, edges

# Function to decide theta value of nodes. Can override for different configurations.
def get_node_theta(i):
    return random.random()

# Function to decide edge probabilities.
def get_edge_prob(i, j):
    return random.random()


# Get the inner node with most good-node edges
def get_highest_deg_seed(G, inner, outer, budgeted, p):

    if budgeted:

        seed = None
        max_deg = 0

        for n in inner:
            count = 0
            for j in G.edges[n]:
                if j in inner:
                    count += 1
            if count > max_deg:
                seed = n
                max_deg = count

        return [n]

    else:
        seeds = []
        for n in inner:
            count = 0
            for j in G.edges[n]:
                if j in inner:
                    count += 1
                else:
                    count -=1

            if count > 0:
                seeds.append(n)

        return seeds

# Returns random seed
def get_random_seed(G, inner, outer, budgeted, p):
    if budgeted:
        rand = int(np.random.uniform(0,1) * len(inner))
        return [inner[rand]]

    else:
        num_seeds = int((1-p) * len(inner))
        indices = random.sample(range(len(inner)), num_seeds)
        seeds = [inner[i] for i in sorted(indices)]
        return seeds
    
# Returns seed with furthest average distance from out-node
def get_furthest_seed(G, inner, outer, budgeted, p):

    if budgeted:
        seed = None

        # Find node that is furthest from bad nodes.
        min_score = 1000
        for n in inner:
            score = 0
            for j in outer:
                if G.distMP2[n][j]:
                    score += 1 / G.distMP2[n][j]
            
            if score < min_score:
                min_score = score
                seed = n

        return [seed]

    else:
        num_seeds = int((1-p) * len(inner))
        heap = []
        for n in inner:
            score = 0
            for j in outer:
                if G.distMP2[n][j]:
                    score += 1 / G.distMP2[n][j]

            heapq.heappush(heap, (score, n))

        seed_pairs = heapq.nsmallest(3, heap)
        seeds = [x[1] for x in seed_pairs]
        return seeds
        

# Gets seed closest to inner and furthest from outer
def get_nearest_seed(G, inner, outer, budgeted, p):
    if budgeted:
        seed = None
        max_score = -1000

        for n in inner:
            score = 0.0

            for j in inner:
                score += 1.0 / G.distMP2[n][j]
        
            for j in outer:
                score -= 1.0 / G.distMP2[n][j]

            if score > max_score:
                max_score = score
                seed = n

        return [seed]

    else:
        seeds = []
        for n in inner:
            score = 0.0
            for j in inner:
                score += 1.0 / G.distMP2[n][j]

            for j in outer:
                score -= 1.0 / G.distMP2[n][j]

            if score > 0:
                seeds.append(n)

        return seeds

def get_btwness_seed(G, inner, outer, budgeted, p):
    if budgeted:
        max_seed = None
        max_score = 0
        for seed in G.btwness:
            if G.btwness[seed] > max_score:
                max_score = G.btwness[seed]
                max_seed = seed

        return [seed]
    else:
        num_seeds = int((1-p) * len(inner))
        heap = []
        for seed in G.btwness:
            heapq.heappush(heap, (G.btwness[seed], seed))

        seed_pairs = heapq.nlargest(num_seeds, heap)
        seeds = [x[1] for x in seed_pairs]
        return seeds


def _get_A_k(A_1, A_prev_k):
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
            

def get_katz_seed(G, inner, outer, budgeted, p):
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
        A_mats[i] = _get_A_k(A_mats[0], A_mats[i - 1])
        
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

    centrality = np.zeros(len(inner))
    for idx, node in enumerate(inner):
        for k in range(iterations):
            try:
                centrality[idx] += np.power(p, k + 1) * np.sum(A_binary[k][idx])
            except Exception as e:
                print "Let Soobin know that he screwed up."
                raise e

    nodes_by_centrality = np.argsort(-centrality)

    if budgeted:
        num_seeds = 1
    else: 
        num_seeds = int((1-p) * len(inner))

    return [inner[nodes_by_centrality[i]] for i in range(num_seeds)]

def get_katz_seed_nx(G, inner, outer, budgeted, p):
    # Set up NX graph
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(G.nodeThetas.keys())
    edges_added = set()
    for pair, prob in G.edgeProb.items():
        if (pair[0], pair[1]) in edges_added or pair[0] in outer or pair[1] in outer:
            continue
        nx_graph.add_edge(pair[0], pair[1], weight = 1.0/prob)
        edges_added.add((pair[0], pair[1]))
        edges_added.add((pair[1], pair[0]))

    katz_by_node = nx.katz_centrality_numpy(nx_graph, alpha = p)
    max_cent = 0
    seed = None
    for node_id, cur_cent in katz_by_node.items():
        if cur_cent > max_cent:
            max_cent = cur_cent
            seed = node_id
    
    nodes_sorted_by_katz = sorted(katz_by_node, key=katz_by_node.get, reverse=True)

    if budgeted:
        num_seeds = 1
    else: 
        num_seeds = int((1-p) * len(inner))

    return [inner[nodes_sorted_by_katz[i]] for i in range(num_seeds)]
            


def run():

    random.seed(0)
    header = ['Theta', 'Inner', 'Outer', 'Ratio', 'Prob', 'Opt',
              'HD', 'Rand', 'Far', 'Near', "Btw", 'katz']
    results = []
    results_budgeted = []

    # Number of iterations to run.
    NUM = 100

    # Construct initial network graph.
    size, nodes, edges = get_enron_data()
    G = graph.Graph(size, nodes, edges)

    for theta in np.linspace(0.1, 0.9, 17):
        graph_file = config.graph_file.format(theta)
        start = time.time()
        if os.path.isfile(graph_file):
            SG = pickle.load(open(graph_file, 'rb'))
        else:
            SG =  G.get_largest_cluster_subgraph(theta)
            SG.get_distance_matrix()
            pickle.dump(SG, open(graph_file, 'wb'))

        end = time.time()

        start = time.time()
        SG.calculate_btwness(theta)
        end = time.time()

        inner, outer = SG.get_inner_outer_nodes(theta)

        print "\n\nTheta: {0}, inner:{1}, outer:{2}".format(theta,
                                                        len(inner),
                                                        len(outer))

        ratio = float(len(inner)) / (len(outer) + len(inner))
        best_possible = len(inner) + len(outer)

        for p in np.linspace(0.1, 0.9, 17):

            def prob_func(i, j):
                return p

            start = time.time()
            SG.set_edge_probabilities(prob_func)
            SG.calculate_distance_matrix_with_prob(p)

            algs = [get_highest_deg_seed, get_random_seed, get_furthest_seed,
                    get_nearest_seed, get_btwness_seed, get_katz_seed]
            
            scores = [0.0] * len(algs)
            scores_budgeted = [0.0] * len(algs)

            start = time.time()
        
            for i in range(NUM):
                G_theta = SG.get_subgraph(theta)
                budgeted = True
                for ind, alg in enumerate(algs):
                    seed = alg(SG, inner, outer, budgeted, p)
                    scores[ind] += G_theta.get_seed_score(seed)

                budgeted = False
                for ind, alg in enumerate(algs):
                    seed = alg(SG, inner, outer, budgeted, p)
                    scores_budgeted[ind] += G_theta.get_seed_score(seed)
                    
            end = time.time()
            print "Calculating Seed: {0}".format((end - start)/60)

            norm_scores = [(x / NUM  + len(outer)) / best_possible for x in scores]
            norm_scores_budgeted = [(x / NUM  + len(outer)) / best_possible
                                    for x in scores_budgeted]

            setting = [theta, len(inner), len(outer), ratio, p, best_possible]
            row = setting + norm_scores
            results.append(row)

            row_b = setting + norm_scores_budgeted
            results_budgeted.append(row_b)
            
    pickle.dump(results, open(config.results_p, 'wb'))
    pickle.dump(results_budgeted, open(config.results_budgeted_p, 'wb'))

    with open(config.results_file, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in results:
            writer.writerow(row)

    with open(config.results_file_budgeted, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in results_budgeted:
            writer.writerow(row)

