from pathlib import Path, PurePath

import pandas as pd, numpy as np, math, matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import logging, os, sys, traceback
from datetime import datetime
from pathlib import Path

BASE_LIMIT  = 600   # seconds for n < 500
LARGE_LIMIT = 1200  # seconds for n >= 500
SIZES = [10,20,30,40,50,70,100,200,500,750,1000]

ROOT      = Path(__file__).parent
INPUT_CSV = ROOT / 'input_files' / 'tsp_locations_1000.csv'
OUT_DIR   = ROOT / 'output_files'
PLOT_DIR  = OUT_DIR / 'plots'
CSV_FILE  = OUT_DIR / 'results.csv'
LOG_DIR   = ROOT / 'logs'

# make required dirs
for p in (OUT_DIR, PLOT_DIR, LOG_DIR):
    p.mkdir(exist_ok=True)

#  Logging 
log_file = LOG_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_tsp.log"
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logger.info("CWD: %s", Path.cwd())

time_limit = lambda n: LARGE_LIMIT if n>=500 else BASE_LIMIT

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*6371*math.asin(math.sqrt(a))

def dist_matrix(df, n):
    sub = df.head(n).reset_index(drop=True)
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            D[i,j] = D[j,i] = haversine(sub.at[i,'Latitude'],sub.at[i,'Longitude'],
                                        sub.at[j,'Latitude'],sub.at[j,'Longitude'])
    return D, sub

def extract_gap(results):
    for attr in ('mip_gap','gap'):
        if hasattr(results.solver, attr):
            v = getattr(results.solver, attr)
            if v is not None:
                return v
    return getattr(results.problem, 'MIPGap', None)

def solve_tsp(D, solver_name, limit):
    n=len(D)
    m=pyo.ConcreteModel()
    m.N=pyo.RangeSet(0,n-1)
    m.d=pyo.Param(m.N,m.N,initialize=lambda _,i,j: float(D[i][j]))
    m.x=pyo.Var(m.N,m.N,within=pyo.Binary)
    m.u=pyo.Var(m.N,bounds=(0,n-1))
    m.obj=pyo.Objective(expr=sum(m.d[i,j]*m.x[i,j] for i in m.N for j in m.N if i!=j))
    m.out=pyo.Constraint(m.N,rule=lambda m,i: sum(m.x[i,j] for j in m.N if j!=i)==1)
    m._in=pyo.Constraint(m.N,rule=lambda m,j: sum(m.x[i,j] for i in m.N if i!=j)==1)
    m.mtz=pyo.ConstraintList()
    for i in range(1,n):
        for j in range(1,n):
            if i!=j:
                m.mtz.add(m.u[i]-m.u[j]+n*m.x[i,j] <= n-1)
    solver=SolverFactory(solver_name)
    if solver_name=='gurobi':
        solver.options['TimeLimit']=limit
        solver.options['MIPGap']=0.0001
    else:
        solver.options['seconds']=limit
        solver.options['ratio']=0.0001
        solver.options['threads']=4
    start=datetime.now()
    try:
        res=solver.solve(m,tee=False)
    except Exception as e:
        logger.error("%s crashed: %s", solver_name, e)
        logger.debug(traceback.format_exc())
        return None
    runtime=(datetime.now()-start).total_seconds()
    arcs=[(i,j) for i in range(n) for j in range(n) if i!=j and pyo.value(m.x[i,j])>0.5]
    dist=sum(D[i][j] for i,j in arcs)
    gap=extract_gap(res)
    if res.solver.termination_condition==pyo.TerminationCondition.optimal: gap=0.0
    status='Optimal' if res.solver.termination_condition==pyo.TerminationCondition.optimal else str(res.solver.termination_condition)
    return {'distance':dist,'runtime':runtime,'mip_gap':gap,'status':status,'tour':arcs}

def plot_route(sub,tour,n,solver):
    if not tour: return
    plt.figure(figsize=(8,6))
    plt.scatter(sub['Longitude'],sub['Latitude'],c='red')
    for idx,row in sub.iterrows():
        plt.annotate(idx,(row['Longitude'],row['Latitude']),textcoords='offset points',xytext=(0,5),ha='center',fontsize=6)
    for i,j in tour:
        plt.plot([sub.at[i,'Longitude'],sub.at[j,'Longitude']],[sub.at[i,'Latitude'],sub.at[j,'Latitude']],alpha=0.5)
    fname=PLOT_DIR/f'tour_{solver}_{n}.png'
    plt.title(f'{solver.upper()} {n}')
    plt.savefig(fname,dpi=120,bbox_inches='tight'); plt.close()
    logger.info('Plot %s', fname)

def main():
    try:
        df=pd.read_csv(INPUT_CSV)
    except Exception as e:
        logger.error('CSV read error: %s', e); return
    cols=['Num_cities','Gurobi_Distance','Gurobi_Runtime','Gurobi_MipGap','Gurobi_Status',
          'CBC_Distance','CBC_Runtime','CBC_MipGap','CBC_Status']
    pd.DataFrame(columns=cols).to_csv(CSV_FILE,index=False)
    rows={n:{'Num_cities':n} for n in SIZES}

    # Gurobi pass
    for n in SIZES:
        D,sub=dist_matrix(df,n)
        res=solve_tsp(D,'gurobi',time_limit(n))
        if res:
            rows[n].update({'Gurobi_Distance':res['distance'],'Gurobi_Runtime':res['runtime'],
                            'Gurobi_MipGap':res['mip_gap'],'Gurobi_Status':res['status']})
            plot_route(sub,res['tour'],n,'gurobi')
        else: rows[n]['Gurobi_Status']='Error'
        pd.DataFrame(rows.values()).to_csv(CSV_FILE,index=False)

    # CBC pass
    for n in SIZES:
        D,sub=dist_matrix(df,n)
        res=solve_tsp(D,'cbc',time_limit(n))
        if res:
            rows[n].update({'CBC_Distance':res['distance'],'CBC_Runtime':res['runtime'],
                            'CBC_MipGap':res['mip_gap'],'CBC_Status':res['status']})
            plot_route(sub,res['tour'],n,'cbc')
        else: rows[n]['CBC_Status']='Error'
        pd.DataFrame(rows.values()).to_csv(CSV_FILE,index=False)

    final=pd.DataFrame(rows.values()); final.to_csv(CSV_FILE,index=False)
    logger.info('Done. Results CSV at %s', CSV_FILE)
    print('\\n=== FINAL RESULTS ==='); print(final.to_string(index=False))

if __name__=='__main__':
    main()