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

def worst_approximated(workload_answers, est, workload, eps, penalty=True):
    """ Select a (noisy) worst-approximated marginal for measurement.
    
    :param workload_answers: a dictionary of true answers to the workload
        keys are cliques
        values are numpy arrays, corresponding to the counts in the marginal
    :param est: a GraphicalModel object that approximates the data distribution
    :param: workload: The list of candidates to consider in the exponential mechanism
    :param eps: the privacy budget to use for this step.
    """
    errors = np.array([])
    for cl in workload:
        bias = est.domain.size(cl) if penalty else 0
        x = workload_answers[cl]
        xest = est.project(cl).datavector()
        errors = np.append(errors, np.abs(x - xest).sum()-bias)
    sensitivity = 2.0
    prob = softmax(0.5*eps/sensitivity*(errors - errors.max()))
    key = np.random.choice(len(errors), p=prob)
    return workload[key]


def mwem_pgm(data_in,epsilon, lastsyn_load, delta=0.0, cliques_o=None,rounds=None, maxsize_mb = 25, pgm_iters=100, noise='laplace'):
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
    - The dynamic update version *not need* workload. In this version, we used the cliques which choosen by original
        data synthetic as workload
    """ 
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
            rounds = len(cliques)+1

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
        cl = worst_approximated(workload_answers = answers, est = lastsyn_load, workload = workload, eps = exp_eps)
        eps_per_round = (epsilon - exp_eps) / (2 * rounds)
        sigma = 1.0 / eps_per_round #IncreSyn: Re-calculate sigma
        cliques.append(cl)

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
    return est.synthetic_data()

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
                    maxsize_mb=args.max_model_size,
                    pgm_iters=args.pgm_iters
                    )
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
