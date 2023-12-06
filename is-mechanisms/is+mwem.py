import numpy as np
import itertools
import pandas as pd
from mbi import Dataset, GraphicalModel, FactoredInference
from scipy.special import softmax
from scipy import sparse
from cdp2adp import cdp_rho
import argparse
import time
import pickle


"""
This file contains a IncreSyn construction example in the update phase.
For more details of Private-PGM and its implemention, please visit
https://github.com/ryan112358/private-pgm

Before using this or any other mechanisms in IncreSyn, make sure you have
already prepared source code of hdmm and mbi for dependences and put the "src" 
folder's path to PYTHONPATH.
"""

def worst_approximated_eta(workload_answers, est, workload, eps, eta, penalty=True):
    """ Select eta (noisy) worst-approximated marginal for measurement.
    
    :param workload_answers: a dictionary of true answers to the workload
        keys are cliques
        values are numpy arrays, corresponding to the counts in the marginal
    :param est: a GraphicalModel object that approximates the data distribution
    :param: workload: The list of candidates to consider in the exponential mechanism
    :param eps: the privacy budget to use for this step.
    :param eta: the number of selected cliques
    """
    errors = np.array([])
    for cl in workload:
        bias = est.domain.size(cl) if penalty else 0
        x = workload_answers[cl]
        xest = est.project(cl).datavector()
        errors = np.append(errors, np.abs(x - xest).sum()-bias)
    sensitivity = 2.0
    prob = softmax(0.5*eps/sensitivity*(errors - errors.max()))
    keys = np.random.choice(len(errors), p=prob,size = eta)
    choice_cl = []
    for key in keys:
        choice_cl.append(workload[key])
    return choice_cl

def eta_test(data_in, lastsyn_load, syn, workload):
    """
    Test function for adaptively increse eta
    data_in: Original data
    lastsyn_load: Last synthetic data
    syn: Synthetic data of this time
    workload: The workload using for test
    """
    errors_last = []
    errors_this = []
    for proj in workload:
        O = data_in.project(proj).datavector()
        X = lastsyn_load.project(proj).datavector()
        Y = syn.project(proj).datavector()
        elast = 0.5*np.linalg.norm(O/O.sum() - X/X.sum(), 1)
        ethis = 0.5*np.linalg.norm(O/O.sum() - Y/Y.sum(), 1)
        errors_last.append(elast)
        errors_this.append(ethis)

    if np.mean(errors_this) < np.mean(errors_last):
        return 1
    else:
        return 0 
    



def mwem_pgm(data_in,epsilon, lastsyn_load, delta=0.0, cliques_o=None,rounds=None, workload = None, maxsize_mb = 25, pgm_iters=100, noise='laplace', eta_max=5):
    """
    Implementation of a dynamic update version of MWEM+PGM

    :param data_in: an *ORIGINAL* mbi.Dataset object
    :param epsilon: privacy budget
    :param delta: privacy parameter (ignored)
    :param cliques_o: A list of cliques (attribute tuples) which choosen in original synthetic mechanism
    :param rounds: The number of rounds of MWEM to run (default: number of attributes)
    :param maxsize_mb: [New] a limit on the size of the model (in megabytes), used to filter out candidate cliques from selection.
        Used to avoid MWEM+PGM failure modes (intractable model sizes).   
        Set to np.inf if you would like to run MWEM as originally described without this modification 
        (Note it may exceed resource limits if run for too many rounds)

    Implementation Notes:
    - During each round of MWEM, one clique will be selected for measurement, but only if measuring the clique does
        not increase size of the graphical model too much
    """ 
    eta = 1 # IncreSyn: Initialzed an eta = 1
    if workload is None:
        workload = list(itertools.combinations(data_in.domain, 2))
    
    answers = { cl : data.project(cl).datavector() for cl in workload } #IncreSyn: Get workload answers

    cliques = []
    cliquepd = pd.read_csv(cliques_o).values.tolist() #IncreSyn:Get selected cliques
    for line in cliquepd:
        cliques.append(tuple(line)) 
    #IncreSyn:Add prefer cliques
    prefer_cliques = []
    #IncreSyn:Load prefer cliques from file
    prefer_pd = pd.read_csv("./data/prefer.csv").values.tolist() #IncreSyn: Get prefer cliques
    for line in prefer_pd:
            prefer_cliques.append(tuple(line))
    #IncreSyn:Add prefer cliques to original cliques
    cliques += prefer_cliques

    if rounds is None:
        if lastsyn_load is None:
            rounds = len(cliques)
        else:
            rounds = len(cliques)+eta

    if noise == 'laplace':
        eps_per_round = epsilon / (2 * rounds)
        sigma = 1.0 / eps_per_round
        exp_eps = eps_per_round
        marginal_sensitivity = 2
    else:
        rho = cdp_rho(epsilon, delta)
        rho_per_round = rho / (2 * rounds)
        sigma = np.sqrt(0.5 / rho_per_round)
        exp_eps = np.sqrt(8 * rho_per_round)
        marginal_sensitivity = np.sqrt(2)

    domain = data_in.domain
    total = data_in.records
    def size(cliques):
        return GraphicalModel(domain, cliques).size * 8 / 2**20
    engine = FactoredInference(data_in.domain, log=False, iters=pgm_iters, warm_start=True)
    measurements = []
    time_start = time.time()

     #IncreSyn: When the last synthetic data is given, run the select step once
    if lastsyn_load is not None: 
        print('Last synthetic data detected, adding selection')
        choice_cl = worst_approximated_eta(workload_answers = answers, est = lastsyn_load, workload = workload, eta=eta, eps = exp_eps)

        for cl in choice_cl:
            if cl not in cliques:
                cliques.append(cl)
            else:
                rounds = rounds-1
        
        eps_per_round = (epsilon - exp_eps) / (2 * rounds)
        sigma = 1.0 / eps_per_round #IncreSyn: Re-calculate sigma

    for i in range(1, rounds+1):
        ax = cliques[i-1] #IncreSyn: Switch the original select method to reading selected cliques line by line.
        print('Round', i, 'Selected', ax, "Eps per round =",eps_per_round)
        n = domain.size(ax)
        x = data_in.project(ax).datavector()
        if noise == 'laplace':
            y = x + np.random.laplace(loc=0, scale=marginal_sensitivity*sigma, size=n)
        else:
            y = x + np.random.normal(loc=0, scale=marginal_sensitivity*sigma, size=n)
        Q = sparse.eye(n)
        measurements.append((Q, y, 1.0, ax))
    est = engine.estimate(measurements, total) #IncreSyn: Move the estimation outside of the iteration.
    time_end = time.time()
    time_consume=int(round((time_end-time_start) * 1000))
    print('Time cost:'+str(time_consume)+' ms. Saving model...')
    print('Generating Data...')
    syn = est.synthetic_data()
    if lastsyn_load is not None:
        error_comp = eta_test(data_in=data_in, lastsyn_load=lastsyn_load, syn=syn, workload=workload) #IncreSyn:Test for whether eta should gets bigger or not
        if (error_comp == 1) and eta < eta_max:
            eta +=1
            print("Eta increased to "+str(eta))
    return syn

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = '../data/colorado.csv'
    params['domain'] = '../data/colorado-domain.json'
    params['epsilon'] = 1.0
    params['delta'] = 1e-9
    params['cliques'] = '../data/cliques.csv'
    params['model'] = '../data/models'
    params['rounds'] = None
    params['noise'] = 'laplace'
    params['max_model_size'] = 25
    params['pgm_iters'] = 250
    params['degree'] = 2
    params['num_marginals'] = None
    params['max_cells'] = 10000
    params['lastsyn'] = None
    params['eta'] = 5

    return params

if __name__ == "__main__":

    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', help='Modified dataset to use')
    parser.add_argument('--domain', help='domain to use')
    parser.add_argument('--epsilon', type=float, help='privacy parameter')
    parser.add_argument('--delta', type=float, help='privacy parameter')
    parser.add_argument('--cliques',help='Cliques to use')
    parser.add_argument('--rounds', type=int, help='number of rounds of MWEM to run')
    parser.add_argument('--noise', choices=['laplace','gaussian'], help='noise distribution to use')
    parser.add_argument('--max_model_size', type=float, help='maximum size (in megabytes) of model')

    parser.add_argument('--degree', type=int, help='degree of marginals in workload')
    parser.add_argument('--num_marginals', type=int, help='number of marginals in workload')
    parser.add_argument('--max_cells', type=int, help='maximum number of cells for marginals in workload')

    parser.add_argument('--pgm_iters', type=int, help='number of iterations')
    parser.add_argument('--save', type=str, help='path to save synthetic data')
    parser.add_argument('--lastsyn', help = 'last synthetic data')
    parser.add_argument('--eta', type=int, help = 'Threshold of eta')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data = Dataset.load(args.dataset, args.domain)

    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        workload = [workload[i] for i in prng.choice(len(workload), args.num_marginals, replace=False)]
    
    if args.lastsyn is not None:
        lastsyn_load = Dataset.load(args.lastsyn, args.domain)
    else:
        lastsyn_load = None

    synth = mwem_pgm(data, args.epsilon,lastsyn_load, args.delta, 
                    cliques_o=args.cliques,
                    rounds=args.rounds,
                    workload=workload,
                    maxsize_mb=args.max_model_size,
                    pgm_iters=args.pgm_iters,
                    eta_max = args.eta)
    if args.save is not None:
        synth.df.to_csv(args.save, index=False)

    errors = []
    for proj in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors.append(e)
    print('Average Error: ', np.mean(errors))
    #IncreSyn: Calc prefer attributes error
    prefer_cliques = []
    prefer_pd = pd.read_csv("./data/prefer.csv").values.tolist()
    for line in prefer_pd:
        prefer_cliques.append(tuple(line))
    errors_p = []
    for proj in prefer_cliques:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors_p.append(e)
    print('Average Error in Preferred Cliques: ',np.mean(errors_p))
