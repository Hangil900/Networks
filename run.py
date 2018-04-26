import graph, config, pdb
import random, csv, pickle
import numpy as np
from subgraph import Subgraph
#import plot

random.seed(0)

def get_enron_data():

    edges = []
    nodes = set()
    
    with open(config.enron_email_data, 'rb') as f:
        lines = f.readlines()

        for line in lines[4:]:
            lineA = line.split()
            i = int(lineA[0])
            j = int(lineA[1])

            edges.append((i, j, 1))
            nodes.add(i)
            nodes.add(j)

    if len(nodes) - 1 != max(nodes):
        pdb.set_trace()

    return edges, len(nodes)

# Get the inner node with most in-node edges
def get_highest_deg_seed(G, inner, outer):

    seed = None
    max_deg = 0

    for n in inner:
        count = 0
        for j in G.M[n]:
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
def get_furthest_seed(G, inner, outer, p):
    seed = None
    min_score = 1000000000

    for n in inner:
        score = 0
        for j in outer:
            score += (p ** G.distM[n][j])

        if score < min_score:
            min_score = score
            seed = n

    return seed

# Gets seed closest to inner and furthest from outer
def get_nearest_seed(G, inner, outer, p):
    seed = None
    max_score = 0.0

    for n in inner:
        score = 0.0

        for j in inner:
            score += (p **  G.distM[n][j])
        
        for j in outer:
            score -= (p ** G.distM[n][j])

        if score > max_score:
            max_score = score
            seed = n

    return seed

def run():

    header = ['Theta', 'Ratio', 'Prob', 'HD', 'Rand', 'Far', 'Near']
    results = []

    NUM = 100


    edges, size = get_enron_data()
    G = graph.Graph(edges, size, 1)

    for theta in np.linspace(0.1, 0.9, 17):
        cluster  = G.get_largest_cluster(theta)
        inner, outer = (cluster[0]), (cluster[1])
        print "\n\nTheta: {0}, inner:{1}, outer:{2}".format(theta,
                                                        len(inner),
                                                        len(outer))

        ratio = float(len(inner)) / len(outer)

        for p in np.linspace(0.1, 0.9, 17):

            highest_deg_score = 0.0
            random_score = 0.0
            furthest_score = 0.0
            nearest_score = 0.0
        
            for i in range(NUM):
                G_theta = G.get_subgraph(cluster, p, theta)

                highest_deg_seed = get_highest_deg_seed(G,inner, outer)
                random_seed = get_random_seed(G, inner, outer)
                furthest_seed = get_furthest_seed(G, inner, outer,p )
                nearest_seed = get_nearest_seed(G, inner, outer, p)

                highest_deg_score += G_theta.get_seed_score(highest_deg_seed)
                random_score += G_theta.get_seed_score(random_seed)
                furthest_score += G_theta.get_seed_score(furthest_seed)
                nearest_score += G_theta.get_seed_score(nearest_seed)


            highest_deg_score /= NUM
            random_score /= NUM
            furthest_score /= NUM
            nearest_score /= NUM

            row = [theta, ratio, p, highest_deg_score, random_score,
                   furthest_score, nearest_score]

            results.append(row)
            

    pickle.dump(results, open(config.results_p, 'wb'))

    with open(config.results_file, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in results:
            writer.writerow(row)

#def plot():
#    plot.plot_results()


    
"""
def test():
    edges = [(0,1,1),
             (0,2,1),
             (0,6,1),
             (1,3,1),
             (1,4,1),
             (2,5,1),
             (7,8,1)]

    num_nodes = 10
    inner = set([0,1,2,6,7,9])
    outer = set([3,4,5,8])

    sg = Subgraph(edges, num_nodes, inner, outer)

    for i in range(10):
        print "Seed: {0}, Score:{1}".format(i, sg.get_seed_score(i))
"""
