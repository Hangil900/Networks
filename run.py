
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
            


def run():

    random.seed(0)

    header = ['Theta', 'Ratio', 'Prob', 'HD', 'Rand', 'Far', 'Near', "Btw"]
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
        print "Calculating Graph: {0}".format((start - end)/60)

        start = time.time()
        SG.calculate_btwness(theta)
        end = time.time()
        print "Calculating btwness: {0}".format((start - end)/60)

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
            end = time.time()
            print "Calculating distMP: {0}".format((start - end)/60)

            algs = [get_highest_deg_seed, get_random_seed, get_furthest_seed,
                    get_nearest_seed, get_btwness_seed]
            
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
            print "Calculating Seed: {0}".format((start - end)/60)

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

