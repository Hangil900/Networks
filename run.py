
import graph, config, pdb
import random, csv, pickle
import numpy as np
import os
import time, heapq
import plot

# Parses enron data file.
def get_enron_data():

    edges = []
    nodes = []
    edge_set = set()
    node_set = set()
    
    with open(config.enron_email_data, 'rb') as f:
        lines = f.readlines()

        for line in lines[4:]:
            lineA = line.split()
            i = int(lineA[0])
            j = int(lineA[1])

            theta1 = get_node_theta(i)
            theta2 = get_node_theta(j)
            p = get_edge_prob(i, j)

            edge = (i,j)
            edge2 = (j,i)

            if edge not in edge_set:
                edge_set.add(edge)
                edges.append((i, j, p) )

            if edge2 not in edge_set:
                edge_set.add(edge2)
                edges.append( (j,i, p) )

            if i not in node_set:
                node_set.add(i)
                nodes.append((i, theta1))

            if j not in node_set:
                node_set.add(j)
                nodes.append((j , theta2))

    assert len(nodes) == len(node_set)
    assert len(edges) == len(edge_set)
    print len(nodes), len(edges)
    
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
        num_seeds = int((1-p) * len(inner))
        heap = []
        for n in inner:
            score = 0
            for j in G.edges[n]:
                if j in inner:
                    score += 1
                else:
                    score -=1
            heapq.heappush(heap, (score, n))

        seed_pairs = heapq.nsmallest(num_seeds, heap)
        seeds = [x[1] for x in seed_pairs]
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

        seed_pairs = heapq.nsmallest(num_seeds, heap)
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
    A_binary, inner, outer, iterations = G.katz_vars
    centrality = np.zeros(len(inner))
    for idx, node in enumerate(inner):
        for k in range(iterations):
            try:
                centrality[idx] += np.power(p, k + 1) * np.sum(A_binary[k][idx])
            except Exception as e:
                print "Let Soobin know that he screwed up."
                raise e

    nodes_by_centrality = np.argsort(-centrality)

    seeds = []
    if budgeted:
        seeds.append(inner[nodes_by_centrality[0]])
    else: 
        for ind in nodes_by_centrality:
            if centrality[ind] > 0:
                seeds.append(inner[ind])
            else:
                break
    return seeds
            
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
    all_results = []
    all_results_budgeted = []

    # Number of iterations to run.
    NUM = 100

    # Construct initial network graph.
    size, nodes, edges = get_enron_data()
    G = graph.Graph(size, nodes, edges)

    theta_list =  [0.1, 0.2, 0.4, 0.6, 0.5, 0.3, 0.7, 0.8, 0.9]
    #theta_list = [0.6, 0.5]
    #theta_list = [0.3, 0.7]
    #theta_list = [0.8, 0.9]
    for theta in theta_list:
        theta_results_file = config.theta_results_p.format(theta)
        theta_results_file_budgeted = config.theta_results_budgeted_p.format(theta)

        if (os.path.isfile(theta_results_file) and
            os.path.isfile(theta_results_file_budgeted)):

            results = pickle.load(open(theta_results_file, 'rb'))
            results_budgeted = pickle.load(open(theta_results_file_budgeted, 'rb'))

            all_results.extend(results)
            all_results_budgeted.extend(results_budgeted)

            # Plot results
            plot.plot_results(results, True)
            plot.plot_results(results_budgeted, False)

            continue

        
        results = []
        results_budgeted = []
        graph_file = config.graph_file.format(theta)
        start = time.time()
        if os.path.isfile(graph_file):
            SG = pickle.load(open(graph_file, 'rb'))
        else:
            start = time.time()
            SG =  G.get_largest_cluster_subgraph(theta)
            SG.get_distance_matrix()
            SG.calculate_btwness(theta)
            SG.calculate_katz(theta)
            pickle.dump(SG, open(graph_file, 'wb'))
            end = time.time()
            print "\n\nPrep time: {0}".format( round( (end - start) / 60.0, 3))


        inner, outer = SG.get_inner_outer_nodes(theta)

        print "Theta: {0}, inner:{1}, outer:{2}".format(theta,
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

            seeds = []
            seeds_budgeted = []

            budgeted= True
            for ind, alg in enumerate(algs):
                seed = alg(SG, inner, outer, budgeted, p)
                seeds.append(seed)

            budgeted = False
            for ind, alg in enumerate(algs):
                seed = alg(SG, inner, outer, budgeted, p)
                seeds_budgeted.append(seed)

            for i in range(NUM):
                G_theta = SG.get_subgraph(theta)
                for ind, alg in enumerate(algs):
                    seed = seeds[ind]
                    seed_b = seeds_budgeted[ind]
                    scores[ind] += G_theta.get_seed_score(seed)
                    scores_budgeted[ind] += G_theta.get_seed_score(seed_b)

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

        # Add Theta results
        all_results.extend(results)
        all_results_budgeted.extend(results_budgeted)

        # Pickle Theta results
        pickle.dump(results, open(theta_results_file, 'wb'))
        pickle.dump(results_budgeted,
                    open(theta_results_file_budgeted, 'wb'))

        # Plot results
        plot.plot_results(results, True)
        plot.plot_results(results_budgeted, False)


    print theta_list
    pdb.set_trace()
    
    pickle.dump(all_results, open(config.results_p, 'wb'))
    pickle.dump(all_results_budgeted, open(config.results_budgeted_p, 'wb'))

    with open(config.results_file, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in all_results:
            writer.writerow(row)

    with open(config.results_file_budgeted, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in all_results_budgeted:
            writer.writerow(row)

