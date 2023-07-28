import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# Notice: you needs to pass actully table name to this function, like colorado_cliques 
def clique_read(db_conn, tablename):
    clique_sql = "SELECT * FROM \""+tablename+"\""
    clique_pd =  pd.read_sql(sql=clique_sql, con=db_conn)
    cliques_tuple = []
    for index, row in clique_pd.iterrows():
        cliques_tuple.append(tuple(row))
    return cliques_tuple

def clique_save(db_conn, tablename, cliques_in):
    # We must re-define an engine here
    engine = create_engine(db_conn)
    cliques_pd = pd.DataFrame(cliques_in,columns=None)
    cliques_pd.to_sql(name=str(tablename)+'_cliques', con=engine,index=False)

    
    

