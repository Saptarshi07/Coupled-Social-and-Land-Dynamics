'''
The following script performs fitting to estimate the parameters kappa, beta and alpha of the replicator equation for a country over a parameterization period that begins at variable start and ends at variable end. The user can input three parameters in-line (see below)
'''


#load the necessary packages
import pandas as pd
from scipy import stats
import numpy as np
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
import math
import sys
from scipy.optimize import curve_fit
import csv


#input the FAO country code for the concerned country, the start, and the end years of the parameterization period
country,start_param,end_param = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])


#final finalname of the csv where results will be saved
filename = 'results/param_fit_%d_'%country



#data for global land use from 1961-2013 due to food consumption, calculated from model in Rizvi et al. For more details see Methods and Supplementary Information.

LU = {1961: 2634795966.5178075,
 1962: 2664780297.6079574,
 1963: 2692289049.4274087,
 1964: 2695644970.092953,
 1965: 2724137678.9858847,
 1966: 2722732789.571131,
 1967: 2734834883.4285026,
 1968: 2746318101.6737394,
 1969: 2783806551.6661053,
 1970: 3006049912.8625665,
 1971: 2984558091.664594,
 1972: 3075011199.17755,
 1973: 3062110126.412742,
 1974: 3104793241.967225,
 1975: 3145533120.5795536,
 1976: 3153889869.6599383,
 1977: 3172344286.6078415,
 1978: 3206818438.2534547,
 1979: 3213853724.122739,
 1980: 3308241249.0506163,
 1981: 3290413069.0807843,
 1982: 3344328436.5987597,
 1983: 3396954577.0563765,
 1984: 3355528520.739557,
 1985: 3361673884.1449676,
 1986: 3432763686.7125893,
 1987: 3421660585.39052,
 1988: 3484769603.799239,
 1989: 3476521206.137939,
 1990: 3472401188.003926,
 1991: 3503190543.508094,
 1992: 3519702548.4345,
 1993: 3534834178.543163,
 1994: 3534708802.7146335,
 1995: 3577281687.253833,
 1996: 3554529478.8080783,
 1997: 3592230571.8811207,
 1998: 3588807682.8930793,
 1999: 3599011300.608892,
 2000: 3645857954.9816337,
 2001: 3585453770.128444,
 2002: 3434608131.301626,
 2003: 3626871260.280846,
 2004: 3590515841.175412,
 2005: 3604894697.8315716,
 2006: 3613152507.3890467,
 2007: 3635169473.781164,
 2008: 3581487095.665802,
 2009: 3553274517.0540595,
 2010: 3571649771.1610765,
 2011: 3624343821.5168967,
 2012: 3616369408.458761,
 2013: 3525102708.182174}

#normalizing the global land use time-series
total_LU = {k: LU[k]/max(list(LU.values())) for k in LU}

#load some datasets necessary for parameterization

#dataframe containing cumax, cs and cl for all countries for which parameterization is possible (these are calculated with model developed in Rizvi et al. For details see Methods and Supplementary Information)
cumaxclcs = pd.read_csv('loaddata/cumaxclcs.csv') 

#data for caloric consumption of meat and dairy subgroups for countries between 1961-2013 (FAO dataset), income data for countries (in 2005 USD)
conspd = pd.read_csv('loaddata/meat_consumption_gdp.csv',encoding='latin-1')

#parameters for the sigmoid fitted on the available data for population and income of countries between 1961-2013
sigmoid_coeff = pd.read_csv('loaddata/sigmoid_coeff.csv')

#dataframe for avaialble poverty data for countries (for references see Data availability & Supplementary Information)
poverty = pd.read_csv('loaddata/poverty.csv')


#defining necessary functions for the parameter estimation method


def get_income_conspd(i,t):
    #read the income data for country i at year t from dataframe and returns the normalized value of it. 
    
    t = int(t)
    max_ = max(list(conspd.loc[(conspd['country code'] == i)]['income value']))
    return conspd.loc[(conspd['country code'] == i)&(conspd['year'] == t)]['income value'].values[0]/max_

def get_income_proj(i,t):
    #returns the interpolated value for income (in 2005 USD) from data for country i at year t. The fitting is performed with a sigmoid function. Parameters for the sigmoid are read from the sigmoid_coeff dataframe. 
    
    ret = sigmoid_coeff.loc[(sigmoid_coeff['country code'] == i)&(sigmoid_coeff['element'] == 'income')]
    L, x0, k, b = float(ret['L'].values[0]), float(ret['x0'].values[0]), float(ret['k'].values[0]),  float(ret['b'].values[0])
    return sigmoid_L(t,L,x0,k,b)

def sigmoid_L(x, L ,x0, k, b):
    #defines the sigmoid function with four parameters
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def get_global_LU(year):
    #returns global land use from the dictionary defined earlier for ONLY years between 1961-2013
    
    if(year > 2013):
        return total_LU[2013]
    else:
        year = math.floor(year)
        return total_LU[year]
    
def get_pop_proj(i,t):
    #returns the interpolated value for population from data for country i at year t. The fitting is performed with a sigmoid function. Parameters for the sigmoid are read from the sigmoid_coeff dataframe.
    
    ret = sigmoid_coeff.loc[(sigmoid_coeff['country code'] == i)&(sigmoid_coeff['element'] == 'pop')]
    L, x0, k, b = float(ret['L'].values[0]), float(ret['x0'].values[0]), float(ret['k'].values[0]),  float(ret['b'].values[0])
    return sigmoid_L(t,L,x0,k,b)


def get_cs(country,t):
    #reads the cs value of the 'country' at year t. Precomputed and stored in the dataframe cumaxclcs.
    
    reduced = cumaxclcs.loc[(cumaxclcs['country code'] == country)]
    cs_dict = dict(zip(reduced.year, reduced.cs))
    if t > 2013:
        t = 2013
    t = math.floor(t)
    return cs_dict[t]

def get_cl(country,t):
    #reads the cl value of the 'country' at year t. Precomputed and stored in the dataframe cumaxclcs.
    
    reduced = cumaxclcs.loc[(cumaxclcs['country code'] == country)]
    cl_dict = dict(zip(reduced.year, reduced.cl))
    if t > 2013:
        t = 2013
    t = math.floor(t)
    return cl_dict[t]

def get_cdata(country,t):
    #reads the cdata (per capita land-consumption) value of the 'country' at year t. Precomputed and stored in the dataframe cumaxclcs.
    
    reduced_ = cumaxclcs.loc[(cumaxclcs['country code'] == country)]
    data_dict = dict(zip(reduced_.year, reduced_.used))
    if t > 2013:
        t = 2013
    t = math.floor(t)
    return data_dict[t]



def sigmoid(k,x,x0):
    #defining a sigmoid with 3 parameters
   
    return (1 / (1 + np.exp(-k*(x-x0))))

def get_poverty(country,year):
    #returns the interpolated or extrapolated fraction (not percentage) of the population that is under poverty. We use a sigmoid to interpolate or extrapolate values. Data collected from dataframe poverty
    
    
    red_ = poverty.loc[poverty['country code'] == country][['Year', 'Poverty']]
    save_ = dict(zip(red_.Year, red_.Poverty))
    xdata,ydata = np.array(list(save_.keys())), np.array(list(save_.values()))
    popt, pcov = curve_fit(sigmoid, xdata, ydata, method='dogbox')
    return sigmoid(year,popt[0],popt[1])/100
    
    
def adjusted_consumption(country,year):
    #calculates cA for a 'country' at a 'year' whose poverty level can be interpolated or extrapolated with get_poverty(..)
    
    p = get_poverty(country,year)
    reduced_ = cumaxclcs.loc[(cumaxclcs['country code'] == country)]
    data_used = dict(zip(reduced_.year, reduced_.used))
    cdata = get_cdata(country,year)
    cs = get_cs(country,year)
    
    return (cdata - p*cs)/(1-p)
    
def get_x_data(country,year):
    #use the evaluated cA for a country using adjusted_consumption(...) and cl for a country using get_cl(...) to calculate the model estimated data for x (proportion of population above poverty consuming the eco-conscious diet) for a country at a year. 
    
    cl = get_cl(country,year)
    ca = adjusted_consumption(country,year)
    return 1 - np.exp(-cl/ca)


def xde(x,t,paras):
    #defines the central replicator equation 
    
    i = paras['i'].value
    kappa = paras['kappa'].value
    alpha = paras['alpha'].value
    beta = paras['beta'].value

    l_ = get_global_LU(t)
    max_ = np.nanmax(conspd.loc[conspd['country code'] == i]['income value'])
    if t < 2014:
        m = get_income_conspd(i,t)
        if math.isnan(m):
            m = get_income_proj(i,t)/max_
    else:
        m = get_income_proj(i,t)/max_

    

    op = kappa*x*(1 - x)*(l_ + beta*m + alpha)
    return op

def sol_xde(T,paras):
    #provides a solution of the replicator equation over T for a initial condition x0 which is taken to be the first data point of xi at start.

    x0 = paras['x0'].value
    
    
    sol = odeint(xde, x0, T, args=(paras,))
    return sol

def residual(paras, t, data):

    #compute the absolute residual vector between model estimated data for x and its prediction from the replicator equation
    
    model = sol_xde(t, paras)
    
    weight = []
    N = len(data)
        
    #non-weighted 
    return [abs(model[i] - data[i]) for i in range(0,len(model))]
    #weighted
    #return [abs((model[i] - data[i])*(i+1)/N) for i in range(0,len(model))]
    
def residual_sum(paras, t, data):
    #computing error - summing all elements of residual vector and dividing by size of time-series
    
    N = len(data)
    return sum(residual(paras, t, data))[0]/N



start,end =  start_param,end_param 

t = np.linspace(start,end,end-start+1)


data = [get_x_data(country,t_) for t_ in t] 


#fitting using lmfit package in python

params = Parameters()
params.add('i', value=country, vary=False)
params.add('x0', value=data[0], vary=False)

params.add('kappa',min= 0.01, max = 1, vary = True)
params.add('alpha',min= -1, max = 1,vary = True)
params.add('beta', min= -1,max = 1, vary = True)

result = minimize(residual, params, args=(t, data), method = 'lsq') 

kappa_fit = result.params['kappa'].value
alpha_fit = result.params['alpha'].value
beta_fit = result.params['beta'].value
error_fit = residual_sum(result.params, t, data)


#saving the results in the csv with filename and path defined earlier
csvData = [['country', 'x0', 'start', 'end', 'kappa', 'alpha', 'beta', 'error'], [country, data[0], start, end, kappa_fit, alpha_fit, beta_fit, error_fit]]



with open( filename + '.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)
csvFile.close()
    

