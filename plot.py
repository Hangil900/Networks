import config
import pdb, os
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def update_results():

    results = []

    with open(config.results_file, 'rb') as f:
        lines=f.readlines()
        for line in lines:
            lineA = line.split(',')
            lineA = [float(x) for x in lineA]
            results.append(lineA)

    pickle.dump(results, open(config.results_p, 'wb'))

    inner = [80, 132, 169, 213, 265, 323, 364, 396, 457,
             514, 574, 620, 676, 723, 781, 838, 898]
    
    outers = [681, 692, 680, 649, 607, 572, 540, 513,
              466, 416, 371, 331, 282, 239,
              187, 139, 85]

    results = pickle.load(open(config.results_p, 'rb'))

    
    curr = results[0][0]
    ind = 0

    new_results = []

    for row in results:
        new_row = []
        if row[0] != curr:
            ind += 1
            curr = row[0]

        new_row.extend(row[:3])

        for i in range(3, 7):
            score = float(row[i] + outers[ind]) / (inner[ind] + outers[ind])
            new_row.append(score)

        new_results.append(new_row)

    pickle.dump(new_results, open('./new_results.p', 'wb'))
        

def get_results(algs):
    results = pickle.load(open(config.results_p, 'rb'))
    res_dict = {}
    thetas = set()
    ps = set()

    for row in results:
        thetas.add((row[0], row[1]))
        ps.add(row[2])

    ps = list(ps)
    ps.sort()

    p_indx = {}
    for i, p in enumerate(ps):
        p_indx[p] = i

    for theta in thetas:
        res_dict[theta] = {}
        for alg in algs:
            res_dict[theta][alg] = [None] * len(ps)
            
    for row in results:
        theta = (row[0], row[1])
        p = row[2]
        p_ind = p_indx[p]
        best_possible = float(row[-1])
        off_set = 3

        for i, alg in enumerate(algs):
            res_dict[theta][alg][p_ind] = row[i + off_set] / best_possible

    return res_dict, ps



def plot_results():
    algs = [ 'High-Deg', 'Random', 'Farthest', 'Near-Far', 'Btwness']
    results, x_axis = get_results(algs)
    markers = [ 'v', 's', '>' , '*', '^']
    colors = ['b', 'g', 'y', 'r', 'orange']


    for theta in results:

        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 45}

        matplotlib.rc('font', **font)
        
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.figure(figsize=(30,20))

        for alg, color, marker in zip(algs, colors, markers):
            y_axis = results[theta][alg]

            plt.plot(x_axis, y_axis, color = color, marker = marker,
                     linestyle= 'solid', linewidth = 8,
                     markersize = 15, label=alg)

            plt.ylabel(r"$\delta_r$ - Improvement in Objective")

        plt.title(r"Stochastic Payoff: $\Theta = {0}$, Interior/Exterior = {1}".format(theta[0], round(theta[1], 2)), fontsize = 60)
        plt.xlabel("Propogation Probability", fontsize = 60)
        plt.ylabel("Payoff", fontsize = 60)
        plt.legend(loc = 'upper right')
        
        filename = config.plot_folder + 'theta={0}.png'.format(theta[0])
        plt.savefig(filename)
        plt.close()

    
    
    
    
    
