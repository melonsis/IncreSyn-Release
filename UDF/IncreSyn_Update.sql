-- We strongly recommend you to delete all annotations before use.
-- For details of implemention, see files in is-mechanisms and mechanisms.
DROP FUNCTION IF EXISTS public.incresyn_update(text, real);
CREATE OR REPLACE FUNCTION public.incresyn_update(
	tablename text,
	epsilon real)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$

import numpy as np
import itertools
import pandas as pd
from mbi import Dataset, GraphicalModel, FactoredInference
from scipy.special import softmax
from scipy import sparse
import argparse
import time
from mechanisms.cdp2adp import cdp_rho
from photools.cliques import clique_read
import psycopg2
from sqlalchemy import create_engine

def worst_approximated_eta(workload_answers, est, workload, eps, eta, penalty=True):
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


def mwem_pgm(data_in,epsilon, lastsyn_load,delta=0.0, rounds=None, maxsize_mb = 25, pgm_iters=100, noise='laplace',eta_max=5):

    db_conn = psycopg2.connect(database="fill", user="with", password="Yourown")
    eta = 1
    if workload is None:
        workload = list(itertools.combinations(data_in.domain, 2))
    answers = { cl : data.project(cl).datavector() for cl in workload }
    cliques = []
    cliques = clique_read(db_conn, "select_cliques")
    cliques += clique_read(db_conn, "prefer_cliques")

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

    if lastsyn_load is not None: 
        plpy.notice('Last synthetic data detected, adding selection')
        choice_cl = worst_approximated_eta(workload_answers = answers, est = lastsyn_load, workload = workload, eta=eta, eps = exp_eps)

        for cl in choice_cl:
            if cl not in cliques:
                cliques.append(cl)
            else:
                rounds = rounds-1
        
        eps_per_round = (epsilon - exp_eps) / (2 * rounds)
        sigma = 1.0 / eps_per_round
    
    for i in range(1, rounds+1):

        ax = cliques[i-1]
        plpy.notice('Round', i, 'Selected', ax, "Eps per round =",eps_per_round)

        n = domain.size(ax)
        x = data_in.project(ax).datavector()
        if noise == 'laplace':
            y = x + np.random.laplace(loc=0, scale=marginal_sensitivity*sigma, size=n)
        else:
            y = x + np.random.normal(loc=0, scale=marginal_sensitivity*sigma, size=n)
        Q = sparse.eye(n)
        measurements.append((Q, y, 1.0, ax))

    est = engine.estimate(measurements, total)

    time_end = time.time()
    time_consume=int(round((time_end-time_start) * 1000))
    plpy.notice('Time cost:'+str(time_consume)+' ms.')
    plpy.notice('Generating Data...')
    syn = est.synthetic_data()
    if lastsyn_load is not None:
        error_comp = eta_test(data_in=data_in, lastsyn_load=lastsyn_load, syn=syn, workload=workload)
        if (error_comp == 1) and eta < eta_max:
            eta +=1
            plpy.notice("Eta increased to "+str(eta))
    return syn

def default_params():

    params = {}
    params['epsilon'] = 1.0
    params['delta'] = 1e-9
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
    parser.add_argument('--epsilon', type=float, help='privacy parameter')
    parser.add_argument('--delta', type=float, help='privacy parameter')
    parser.add_argument('--rounds', type=int, help='number of rounds of MWEM to run')
    parser.add_argument('--noise', choices=['laplace','gaussian'], help='noise distribution to use')
    parser.add_argument('--max_model_size', type=float, help='maximum size (in megabytes) of model')

    parser.add_argument('--degree', type=int, help='degree of marginals in workload')
    parser.add_argument('--num_marginals', type=int, help='number of marginals in workload')
    parser.add_argument('--max_cells', type=int, help='maximum number of cells for marginals in workload')

    parser.add_argument('--pgm_iters', type=int, help='number of iterations')
    parser.add_argument('--lastsyn', help = 'last synthetic data')
    parser.add_argument('--eta', type=int, help = 'Threshold of eta')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data = Dataset.load(tablename)
    previous_synth = tablename+'_synth'

    data_previous = Dataset.load(previous_synth)
    plpy.notice("Loaded records, checking diff...")
    diff = len(data.df) - len(data_previous.df)

    if diff > 0:
        plpy.notice("Updating "+str(diff)+" records")
        data.df = data.df[len(data_previous.df):]
    else:
        plpy.notice("WARNING: Size of synthetic data is greater than original data! Using full dataset...")


    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        workload = [workload[i] for i in prng.choice(len(workload), args.num_marginals, replace=False)]
    plpy.notice("Starting update.")

    synth = mwem_pgm(data, args.epsilon, data_previous, args.delta, 
                    rounds=args.rounds,
                    maxsize_mb=args.max_model_size,
                    pgm_iters=args.pgm_iters, eta_max=eta)

    connection = 'postgresql+psycopg2://Fill:With@localhost:5432/yourown'
    engine= create_engine(connection)
    synth.df.to_sql(name=str(tablename)+'_synth', con=engine, index=False, if_exists = 'append') 

    errors = []
    for proj in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors.append(e)
    plpy.notice('Average Error: ', np.mean(errors))

    prefer_cliques = clique_read(db_conn, "prefer_cliques")
    errors_p = []
    for proj in prefer_cliques:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors_p.append(e)
    plpy.notice('Average Error in preferred Cliques: ',np.mean(errors_p))
        $BODY$;

ALTER FUNCTION public.incresyn_update(text, real)
    OWNER TO test;
