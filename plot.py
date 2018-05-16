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
        

def get_results(algs, results):
    res_dict = {}
    thetas = set()
    ps = set()

    for row in results:
        thetas.add((row[0], row[3]))
        ps.add(row[4])

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
        theta = (row[0], row[3])
        p = row[4]
        p_ind = p_indx[p]
        off_set = 6

        for i, alg in enumerate(algs):
            res_dict[theta][alg][p_ind] = row[i + off_set]

    return res_dict, ps



def plot_results(results, budgeted):
    algs = config.algs
    results, x_axis = get_results(config.algs, results)
    markers = [ 'v', 's', '>' , '*', '^', 'd']
    colors = ['b', 'g', 'y', 'r', 'orange', 'pink']


    for theta in results:

        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 40}

        matplotlib.rc('font', **font)
        
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.figure(figsize=(30,20))

        for alg, color, marker in zip(algs, colors, markers):
            y_axis = np.array(results[theta][alg])
            if alg == 'Near-Far':
                y_axis = y_axis + (np.random.random(len(y_axis)) / 50)

            plt.plot(x_axis, y_axis, color = color, marker = marker,
                     linestyle= 'solid', linewidth = 8,
                     markersize = 15, label=alg)

            #plt.ylabel(r"$\Delta$ - Relative Performance")

        if budgeted:
            plt.title(r"{2} Payoff: $\Theta = {0}$, Positive Targets Ratio = {1}".format(theta[0], round(theta[1], 2), 'Budgeted'), fontsize = 60)

        else:
            plt.title(r"{2} Payoff: $\Theta = {0}$, Positive Targets Ratio = {1}".format(theta[0], round(theta[1], 2), 'Unbudgeted'), fontsize = 60)
            
        plt.xlabel("Propogation Probability", fontsize = 60)
        plt.ylabel("Payoff", fontsize = 60)
        plt.legend(loc = 'lower left')
        plt.ylim((0, 1))

        if budgeted:
            filename = config.plot_folder + 'theta={0}.png'.format(theta[0])
        else:
            filename = config.plot_folder_budgeted + 'theta={0}.png'.format(theta[0])
        plt.savefig(filename)
        plt.close()

    
    
    
    
    
