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
def normalError(workload,dataset,synth):
    errors = []
    for proj in workload:
        X = dataset.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors.append(e)
    print('Average Error: ', np.mean(errors))

def preferError(dataset,synth,prefer_cliques_raw):
    #Start to load some prefered CSV, and evaluate cliques in prefered csv's result
    prefer_cliques = []
    prefer_pd = pd.read_csv(prefer_cliques_raw).values.tolist()
    for line in prefer_pd:
        prefer_cliques.append(tuple(line))
    errors_p = []
    for proj in prefer_cliques:
        X = dataset.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors_p.append(e)
    print('Average Error in Preferred Cliques: ',np.mean(errors_p))




def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = './data/colorado/colorado_less.csv'
    params['domain'] = './data/colorado/colorado_less-domain.json'
    params['synth'] = './data/colorado/colorado_synth.csv'
    params['prefer'] = 0
    params['prefer_cliques'] = './data/colorado/perfer.csv'
    params['degree'] = 2
    params['max_cells'] = 10000

    return params

if __name__ == "__main__":
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', help='Original dataset to use')
    parser.add_argument('--domain', help='domain to use')
    parser.add_argument('--synth', help='Synth dataset to use')
    parser.add_argument('--prefer', type=int, help = 'Add prefer or not')
    parser.add_argument('--degree', help = 'degree of marginals in workload')
    parser.add_argument('--max_cells', type=int, help='maximum number of cells for marginals in workload')
    parser.add_argument('--prefer_cliques', help = 'Prefer cliques')

    parser.set_defaults(**default_params())
    args = parser.parse_args()


    data = Dataset.load(args.dataset, args.domain)
    synth = Dataset.load(args.synth, args.domain)
    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    normalError(workload, data, synth)
    if args.prefer == 1:
        preferError(data,synth,args.prefer_cliques)
