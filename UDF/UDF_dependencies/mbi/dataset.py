import numpy as np
import pandas as pd
import os
import json
from mbi import Domain
import psycopg2

class Dataset:
    def __init__(self, df, domain, weights=None):
        """ create a Dataset object

        :param df: a pandas dataframe
        :param domain: a domain object
        :param weight: weight for each row
        """
        assert set(domain.attrs) <= set(df.columns), 'data must contain domain attributes'
        assert weights is None or df.shape[0] == weights.size
        self.domain = domain
        self.df = df.loc[:,domain.attrs]
        self.weights = weights

    @staticmethod
    def synthetic(domain, N):
        """ Generate synthetic data conforming to the given domain

        :param domain: The domain object 
        :param N: the number of individuals
        """
        arr = [np.random.randint(low=0, high=n, size=N) for n in domain.shape]
        values = np.array(arr).T
        df = pd.DataFrame(values, columns = domain.attrs)
        return Dataset(df, domain)

    @staticmethod
    def load(tablename):
        """ Load data into a dataset object from postgresql        
        """ 
        con = psycopg2.connect(database="fill", user="with", password="your own")
        sql_data_cmd = "select * from \""+tablename+"\""
        sql_domain_cmd = "select * from \""+tablename+'_domain\"'
        df_data = pd.read_sql(sql=sql_data_cmd,con=con)
        df_domain = pd.read_sql(sql=sql_domain_cmd,con=con)
        attrs = df_domain['DOMAIN'].tolist()
        sizes = df_domain['SIZE'].tolist()
        domains = {}
        for attr, size in zip(attrs,sizes):
            domains[attr] = size
        domain = Domain.fromdict(domains)
        con.close()
        return Dataset(df_data, domain)

    
    def project(self, cols):
        """ project dataset onto a subset of columns """
        if type(cols) in [str, int]:
            cols = [cols]
        data = self.df.loc[:,cols]
        domain = self.domain.project(cols)
        return Dataset(data, domain, self.weights)

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)
    
    @property
    def records(self):
        return self.df.shape[0]

    def datavector(self, flatten=True):
        """ return the database in vector-of-counts form """
        bins = [range(n+1) for n in self.domain.shape]
        ans = np.histogramdd(self.df.values, bins, weights=self.weights)[0]
        return ans.flatten() if flatten else ans
    
