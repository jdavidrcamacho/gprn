import numpy as np
np.random.seed(23011990)
import matplotlib.pyplot as plt
#%matplotlib notebook
plt.rcParams['figure.figsize'] = [15, 5]
plt.close('all')

#importar funções de covariancia
from gprn.covFunction import SquaredExponential, Periodic
#medias
from gprn.meanFunction import Constant
#SampleMeanField é para usar em mcmc, OptMeanField uso quando uso scipy.optimize
from gprn.simpleMeanFieldOLD import inference

from time import time
points = [10,50,100,200,300,400,500,600,700,800]
calcTime1, calcTime2, calcTime3, calcTime4= [], [], [], []
for i, j in enumerate(points):
    tt,rv,rverr, bis,biserr = np.loadtxt("/home/camacho/GPRN/data/sun{0}points.txt".format(j), 
                               skiprows = 1, unpack = True, usecols = (0,1,2,3,4))
    val1, val1err, val2, val2err = rv, rverr,bis,biserr
    
    #definindo apenas 1 nodo
    nodes1 = [Periodic(1, 1, 25, 0.1)]
    #definindo 2 nodos
    nodes2 = [Periodic(1, 1, 25, 0.1), SquaredExponential(1,1,0.1)]
    #pesos
    weight1 = [SquaredExponential(1, 1, 0.1)]
    #medias para quanto temos apenas rv
    means1 = [Constant(np.mean(val1))]
    #medias para quando tivermos rvs e bis
    means2 = [Constant(np.mean(val1)), Constant(np.mean(val2))]
    #jitter dos rvs
    jitter1 = [np.std(val1)]
    #jitter dos rvs e dos bis
    jitter2 = [np.std(val1), np.std(val2)]
    
    MU, VAR = None, None
    #GPRN 1 nodo  em 1 dataset (apenas rvs)
    start = time()
    #primeiro criamos a gprn neste caso, o 1 vem de usarmos apenas um nodo 
    GPRN = inference(1, tt, val1, val1err)
    #para dados nodos pesos, means e jitter optVarParams optimiza of variational parameters 
    #dado um certo numero de iteraçoes, neste caso 1 nao optimiza nada e apenas calcula o elbo inicial
    GPRN.optVarParams(nodes1, weight1, means1, jitter1, iterations = 1, mu=MU, var=VAR)
    end = time()
    calcTime1.append(end-start)
    
    #gprn 2 nodos em 1 dataset (apenas rvs)
    start = time()
    GPRN = inference(2, tt, val1, val1err) #agora isto tem de levar 2 para funcionar
    #optVarParams vai fazer a mesam coisa que na GPRN anterior
    GPRN.optVarParams(nodes2, weight1, means1, jitter1, iterations = 1, mu=MU, var=VAR)
    end = time()
    calcTime2.append(end-start)
    
    #gprn 1 nodo em 2 datasets (rvs e bis)
    start = time()
    GPRN = inference(1, tt, val1, val1err, val2, val2err) # leva 1 por usamos apenas 1 nodo
    GPRN.optVarParams(nodes1, weight1, means2, jitter2, iterations = 1, mu=MU, var=VAR)
    end = time()
    calcTime3.append(end-start)
    
    #gprn 2 nodos em 2 datasets (rvs e bis)
    start = time()
    GPRN = inference(2, tt, val1, val1err, val2, val2err) #agora leva 2
    GPRN.optVarParams(nodes2, weight1, means2, jitter2, iterations = 1, mu=MU, var=VAR)
    end = time()
    calcTime4.append(end-start)
    
    
plt.figure()
plt.plot(points, calcTime1, '--^', label='OLD RVs (1 node)')
plt.plot(points, calcTime2, '--^', label='OLD RVs (2 nodes)')
plt.plot(points, calcTime3, '--^', label='OLD RVs and BIS(1 node)')
plt.plot(points, calcTime4, '--^', label='OLD RVs and BIS (2 nodes)')


#importar funções de covariancia
from gprn.covFunction import SquaredExponential, Periodic
#medias
from gprn.meanFunction import Constant
#SampleMeanField é para usar em mcmc, OptMeanField uso quando uso scipy.optimize
from gprn.simpleMeanField import inference

from time import time
points = [10,50,100,200,300,400,500,600,700,800]
calcTime1, calcTime2, calcTime3, calcTime4= [], [], [], []
for i, j in enumerate(points):
    tt,rv,rverr, bis,biserr = np.loadtxt("/home/camacho/GPRN/data/sun{0}points.txt".format(j), 
                               skiprows = 1, unpack = True, usecols = (0,1,2,3,4))
    val1, val1err, val2, val2err = rv, rverr,bis,biserr
    
    #definindo apenas 1 nodo
    nodes1 = [Periodic(1, 1, 25, 0.1)]
    #definindo 2 nodos
    nodes2 = [Periodic(1, 1, 25, 0.1), SquaredExponential(1,1,0.1)]
    #pesos
    weight1 = [SquaredExponential(1, 1, 0.1)]
    #medias para quanto temos apenas rv
    means1 = [Constant(np.mean(val1))]
    #medias para quando tivermos rvs e bis
    means2 = [Constant(np.mean(val1)), Constant(np.mean(val2))]
    #jitter dos rvs
    jitter1 = [np.std(val1)]
    #jitter dos rvs e dos bis
    jitter2 = [np.std(val1), np.std(val2)]
    
    MU, VAR = None, None
    #GPRN 1 nodo  em 1 dataset (apenas rvs)
    start = time()
    #primeiro criamos a gprn neste caso, o 1 vem de usarmos apenas um nodo 
    GPRN = inference(1, tt, val1, val1err)
    #para dados nodos pesos, means e jitter optVarParams optimiza of variational parameters 
    #dado um certo numero de iteraçoes, neste caso 1 nao optimiza nada e apenas calcula o elbo inicial
    GPRN.optVarParams(nodes1, weight1, means1, jitter1, iterations = 1, mu=MU, var=VAR)
    end = time()
    calcTime1.append(end-start)
    
    #gprn 2 nodos em 1 dataset (apenas rvs)
    start = time()
    GPRN = inference(2, tt, val1, val1err) #agora isto tem de levar 2 para funcionar
    #optVarParams vai fazer a mesam coisa que na GPRN anterior
    GPRN.optVarParams(nodes2, weight1, means1, jitter1, iterations = 1, mu=MU, var=VAR)
    end = time()
    calcTime2.append(end-start)
    
    #gprn 1 nodo em 2 datasets (rvs e bis)
    start = time()
    GPRN = inference(1, tt, val1, val1err, val2, val2err) # leva 1 por usamos apenas 1 nodo
    GPRN.optVarParams(nodes1, weight1, means2, jitter2, iterations = 1, mu=MU, var=VAR)
    end = time()
    calcTime3.append(end-start)
    
    #gprn 2 nodos em 2 datasets (rvs e bis)
    start = time()
    GPRN = inference(2, tt, val1, val1err, val2, val2err) #agora leva 2
    GPRN.optVarParams(nodes2, weight1, means2, jitter2, iterations = 1, mu=MU, var=VAR)
    end = time()
    calcTime4.append(end-start)
    
    
plt.title('ELBO times')
plt.plot(points, calcTime1, '-o', label='NEW RVs (1 node)')
plt.plot(points, calcTime2, '-o', label='NEW RVs (2 nodes)')
plt.plot(points, calcTime3, '-o', label='NEW RVs and BIS(1 node)')
plt.plot(points, calcTime4, '-o', label='NEW RVs and BIS (2 nodes)')
plt.ylabel('Time (s)')
plt.xlabel('Number of points')
plt.legend()
plt.grid()
plt.show()