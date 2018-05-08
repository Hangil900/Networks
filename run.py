
import graph, config, pdb
import random, csv, pickle
import numpy as np
import os

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
def get_highest_deg_seed(G, inner, outer):

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

    return n

# Returns random seed
def get_random_seed(G, inner, outer):
    rand = int(np.random.uniform(0,1) * len(inner))
    return inner[rand]

# Returns seed with furthest average distance from out-node
def get_furthest_seed(G, inner, outer):
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

    return seed

# Gets seed closest to inner and furthest from outer
def get_nearest_seed(G, inner, outer):
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

    return seed

# Uses all paths, to get "independent" best expected seed.
# Caps expected value at (-1, 1).
def get_exp_best_seed(G, inner ,outer, p):
    if not G.all_paths:
        G.calculate_all_paths()

    seed_scores = []
    seeds = []
        
    for i in range(self.size):
        seed_score = 0
        for target in self.all_paths[i]:
            target_score = 0
            for path in self.all_paths[i][target]:
                if target in inner:
                    target_score += (p ** path)
                else:
                    target_score -= (p ** path)

            target_score = max(-1, min(target_score, 1))
            seed_score += target_score

        seed_scores.append(seed_score)
        seeds.append(i)

    ind = seed_scores.index(max(seed_scores))
    return seeds[i]
            


def run():

    random.seed(0)

    header = ['Theta', 'Ratio', 'Prob', 'HD', 'Rand', 'Far', 'Near', "Expect"]
    results = []

    # Number of iterations to run.
    NUM = 100

    # Construct initial network graph.
    size, nodes, edges = get_enron_data()
    G = graph.Graph(size, nodes, edges)

    for theta in np.linspace(0.1, 0.9, 17):
        graph_file = config.graph_file.format(theta)
        if os.path.isfile(graph_file):
            SG = pickle.load(open(graph_file, 'rb'))
        else:
            SG =  G.get_largest_cluster_subgraph(theta)
            SG.get_distance_matrix()
            pickle.dump(SG, open(graph_file, 'wb'))

        inner, outer = SG.get_inner_outer_nodes(theta)

        print "\n\nTheta: {0}, inner:{1}, outer:{2}".format(theta,
                                                        len(inner),
                                                        len(outer))

        ratio = float(len(inner)) / len(outer)

        for p in np.linspace(0.1, 0.9, 17):

            def prob_func(i, j):
                return p

            SG.set_edge_probabilities(prob_func)
            SG.calculate_distance_matrix_with_prob(p)

            highest_deg_score = 0.0
            random_score = 0.0
            furthest_score = 0.0
            nearest_score = 0.0
            expected_score = 0.0
        
            for i in range(NUM):
                G_theta = SG.get_subgraph(theta)
                highest_deg_seed = get_highest_deg_seed(SG,inner, outer)
                random_seed = get_random_seed(SG, inner, outer)
                furthest_seed = get_furthest_seed(SG, inner, outer,)
                nearest_seed = get_nearest_seed(SG, inner, outer,)
                #expected_seed = get_exp_best_seed(SG, inner ,outer, p)

                highest_deg_score += G_theta.get_seed_score(highest_deg_seed)
                random_score += G_theta.get_seed_score(random_seed)
                furthest_score += G_theta.get_seed_score(furthest_seed)
                nearest_score += G_theta.get_seed_score(nearest_seed)
                #expected_score += G_theta.get_seed_score(expected_seed)


            highest_deg_score /= NUM
            random_score /= NUM
            furthest_score /= NUM
            nearest_score /= NUM
            expected_score /= NUM

            row = [theta, ratio, p, highest_deg_score, random_score,
                   furthest_score, nearest_score, expected_score]

            results.append(row)
            
    pickle.dump(results, open(config.results_p, 'wb'))

    with open(config.results_file, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in results:
            writer.writerow(row)

    pdb.set_trace()
