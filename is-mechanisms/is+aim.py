import numpy as np
import itertools
from mbi import Dataset, GraphicalModel, FactoredInference, Domain
from mechanism import Mechanism
from collections import defaultdict
from hdmm.matrix import Identity
from scipy.optimize import bisect
import pandas as pd
from mbi import Factor
import argparse
import time 

"""
This file contains a IncreSyn construction example in the update phase.
For more details of Private-PGM and its implemention, please visit
https://github.com/ryan112358/private-pgm

Before using this or any other mechanisms in IncreSyn, make sure you have
already prepared source code of hdmm and mbi for dependences and put the "src" 
folder's path to PYTHONPATH.
"""


def powerset(iterable): # Calculting for powerset
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1,len(s)+1))

def downward_closure(Ws): 
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))

def hypothetical_model_size(domain, cliques): 
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2**20

def compile_workload(workload):
    def score(cl):
        return sum(len(set(cl)&set(ax)) for ax in workload)
    return { cl : score(cl) for cl in downward_closure(workload) }

class AIM(Mechanism):
    def __init__(self,epsilon,delta,lastsyn_load,prng=None,rounds=None,max_model_size=80,structural_zeros={},cliques_in = "./data/cliques.csv"):  
        super(AIM, self).__init__(epsilon, delta, prng)
        self.rounds = rounds
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros
        self.cliques_in = cliques_in
        self.lastsyn = lastsyn

    def worst_approximated(self, candidates, answers, model, eps, sigma):
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2/np.pi)*sigma*model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt) 

    def run(self, data, W):
        rounds = self.rounds or 16*len(data.domain) #IncreSyn: Here we using the original rounds limit, to achieve same 1-way calc budget
    
        cliques = []
        cliquepd = pd.read_csv(self.cliques_in).values.tolist() #IncreSyn: Get selected cliques
        for line in cliquepd:
            if line[1] is np.nan:
                cliques.append((line[0],))
            else:
                cliques.append(tuple(line))
        #IncreSyn:Load prefer cliques from file
        prefer_pd =  pd.read_csv("./data/prefer.csv").values.tolist()
        for line in prefer_pd:
            if line[1] is np.nan:
                    cliques.append((line[0],))
            else:
                    cliques.append(tuple(line))
        
       
       
        
        workload = [cl for cl, _ in W]
        candidates = compile_workload(workload)
        answers = { cl : data.project(cl).datavector() for cl in candidates }

        oneway = [cl for cl in candidates if len(cl) == 1] 
        rho_used = 0

        
        sigma = np.sqrt(rounds / (2*0.9*self.rho))
        time_start = time.time()

        #IncreSyn: When the last synthetic data is given, run the select step once
        if self.lastsyn_load is not None: 
            print('Last synthetic data detected, adding selection')
            epsilon = np.sqrt(8*0.1*self.rho/rounds)
            rho_used += epsilon
            cl = self.worst_approximated(workload, answers, lastsyn_load, epsilon, sigma)
            cliques.append(cl)

        measurements = []
        
        print('Initial Sigma', sigma)
        rho_used += len(oneway)*0.5/sigma**2 
        for cl in oneway:
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma,x.size)
            I = Identity(y.size) 
            measurements.append((I, y, sigma, cl))
        zeros = self.structural_zeros
        engine = FactoredInference(data.domain,iters=1000,warm_start=True,structural_zeros=zeros) 
        model = engine.estimate(measurements)
        t = 0
        terminate = False
        
        remaining = self.rho - rho_used
        # IncreSyn: After the completion of a 1-way measurements, we reset the maximum number of rounds to be equal to the total length of cliques (with prefer attributes), in order to avoid allocating too much budget for 1-way measurements. 
        # Once this is set, the subsequent process can be considered as allocating a fixed budget per round.
        rounds = len(cliques) 
        sigma = np.sqrt(rounds / (2 * remaining)) #IncreSyn: Re-design sigma
        print("!!!Re-design sigma after one-way!")
        print("New sigma:",sigma)

        while t < rounds and not terminate:
            t += 1
            cl = None
            if (self.rho - rho_used <0.5/sigma**2): #IncreSyn: Change the limitation
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2*0.9*remaining))
                # We do not needs epsilon here
                # epsilon = np.sqrt(8*0.1*remaining) 
                terminate = True
            rho_used += 0.5/sigma**2 #IncreSyn: Remove epsilon here
            cl = cliques[t-1]        #IncreSyn: Switch the original select method to reading selected cliques line by line.
            n = data.domain.size(cl)
            Q = Identity(n)
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))

            print('Selected',cl,'Size',n,'Budget Used',rho_used/self.rho)


        print("Total rounds:",t)
        engine.iters = 2500
        model = engine.estimate(measurements) #IncreSyn: Move the estimation outside of the iteration.
        time_end = time.time()
        time_consume=int(round((time_end-time_start) * 1000))
        print('Time cost:'+str(time_consume)+' ms.Saving model, cliques and measurements...')
        print('Generating Data...')
        synth = model.synthetic_data()

        return synth

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
    params['noise'] = 'laplace'
    params['max_model_size'] = 80
    params['degree'] = 2
    params['num_marginals'] = None
    params['max_cells'] = 10000
    params['cliques'] = '../data/cliques.csv'
    params['lastsyn'] = None

    return params
        
if __name__ == "__main__":

    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', help='dataset to use')
    parser.add_argument('--domain', help='domain to use')
    parser.add_argument('--epsilon', type=float, help='privacy parameter')
    parser.add_argument('--delta', type=float, help='privacy parameter')
    parser.add_argument('--max_model_size', type=float, help='maximum size (in megabytes) of model')
    parser.add_argument('--degree', type=int, help='degree of marginals in workload')
    parser.add_argument('--num_marginals', type=int, help='number of marginals in workload')
    parser.add_argument('--max_cells', type=int, help='maximum number of cells for marginals in workload')
    parser.add_argument('--save', type=str, help='path to save synthetic data')
    parser.add_argument('--cliques', help='cliques that used')
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


    workload = [(cl, 1.0) for cl in workload]
    mech = AIM(args.epsilon, args.delta, lastsyn_load, max_model_size=args.max_model_size,cliques_in = args.cliques)
    synth = mech.run(data, workload)

    if args.save is not None: # Synthetic save process
        synth.df.to_csv(args.save, index=False)


    errors = []
    for proj, wgt in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*wgt*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors.append(e)
    print('Average Error: ', np.mean(errors))
    
    #IncreSyn: Calc prefer error
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
    print('Average Error in preferred Cliques: ',np.mean(errors_p))

