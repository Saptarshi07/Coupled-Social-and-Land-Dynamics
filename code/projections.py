'''
The following script performs projection of country-wise land-use projection and projections for proportion of population above poverty consuming the eco-conscious diet under population, income and yield scenarios (input throught in-line). Baseline projections are performed when first three in line arguments are 0 (read as pct_kappa, pct_beta, pct_alpha). Projections for deviated parameters by defining the percentage changes in the parameters in these arguments. Projections are from 2011-2100.
'''


#load the necessary packages

import pandas as pd
from scipy import stats
import numpy as np
import csv
import math
import sys
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import sys
import operator


#input the percetage deviation in the baseline parameters for projections
pct_kappa,pct_beta,pct_alpha = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

#input scenario of population-income and the f-scenario for yield.
#f is a real number between 0 and 1
#scenarios can be one of the following: SSP1, SSP2, SSP3, SSP4, SSP5
scenario, f = sys.argv[4], float(sys.argv[5])


#load the necessary datasets

#parameters that were evaluated through fitting
param_res = pd.read_csv('loaddata/parameters.csv')

#ipcc projections for population and income data
ipcc = pd.read_csv('loaddata/IPCC_Projections.csv')

#country wise land-use calculation till 2013 using model in Rizvi et al.
lu_calc = pd.read_csv('loaddata/lu_calc_new.csv')

#the model evaluated data for cumax, cl and cs for countries for years between 1961-2013. Evaluated using model in Rizvi et al.
cumaxclcs = pd.read_csv('loaddata/cumaxclcs.csv')

#data for caloric consumption of meat and dairy subgroups for countries between 1961-2013 (FAO dataset), income data for countries (in 2005 USD)
conspd = pd.read_csv('loaddata/meat_consumption_gdp.csv',encoding='latin-1')

#parameters for the sigmoid fitted on the available data for population and income of countries between 1961-2013
sigmoid_coeff = pd.read_csv('loaddata/sigmoid_coeff.csv')

#dataframe from FAO mapping countries with their FAO country code, ISO3 code and country groups
areapd = pd.read_csv('loaddata/CountryGroupFBS.csv', encoding = "ISO-8859-1")

#dataframe for avaialble poverty data for countries (for references see Data availability & Supplementary Information)
poverty = pd.read_csv('loaddata/poverty.csv')


#of all the IPCC model projections for population and income we use the OECD Env-Growth Model
MODELS = ['OECD Env-Growth']
scen = scenario
SCENARIO = scen + '_v9_130325'
ipcc = ipcc.loc[ipcc['MODEL'].isin(MODELS)&(ipcc['SCENARIO'] == SCENARIO)]

#USD inflation ratio between 2005 and 2018
ratio_05_18 = 0.7551326955350213


#producing a map of FAO country code and ISO3 code of countries under concern
#note that certain countries are eliminated for projections. For more details see Supplementary Information

cc_iso3_map = {}
dictionary_country = {}
val = 0
for i in list(set(lu_calc['country code'].drop_duplicates()) - set([53,184,118,35,75,70,153,188,151,228,17,15,55,51,62,8,186,83,90,144])):
    iso3_ = areapd.loc[areapd['Country Code'] == i]['ISO3 Code'].drop_duplicates().values[0]
    if iso3_ in list(ipcc['REGION']):
        cc_iso3_map[i] =  iso3_
        dictionary_country[val] = i
        val += 1


#performing cubic spline of the IPCC population and income projections data for interpolation purposes

dict_spline_pop = {}
dict_spline_gdp = {}
years = [int(i) for i in ipcc.columns[17:-10]]
for i in cc_iso3_map:
    ypop = list(ipcc.loc[(ipcc['REGION'] == cc_iso3_map[i])&(ipcc['VARIABLE'] == 'Population')][ipcc.columns[17:-10]].values[0])
    ygdp = list(ipcc.loc[(ipcc['REGION'] == cc_iso3_map[i])&(ipcc['VARIABLE'] == 'GDP|PPP')][ipcc.columns[17:-10]].values[0])
    dict_spline_pop[i] = CubicSpline(years, ypop) ## IN MILLIONS
    dict_spline_gdp[i] = CubicSpline(years, ygdp) ## IN BILLIONS 2005 USD
    

#defining functions
def exp_f(x,a,b):
    return math.exp(a)*(math.exp(b*x))

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def straight_line(x,a,b):
    return a*x + b



def get_poverty(country,year):
    #returns the linear interpolated or extrapolated fraction (not percentage) of population that is under poverty
    
    
    red_ = poverty.loc[poverty['country code'] == country][['Year', 'Poverty']]
    save_ = dict(zip(red_.Year, red_.Poverty))
    xdata,ydata = np.array(list(save_.keys())), np.array(list(save_.values()))/100
    popt, pcov = curve_fit(straight_line, xdata, ydata,method='dogbox')
    return straight_line(year,popt[0],popt[1])


def get_pop_proj(i,t):
    #returns an sigmoidal interpolation of country population between 1961 to 2100
    
    ret = sigmoid_coeff.loc[(sigmoid_coeff['country code'] == i)&(sigmoid_coeff['element'] == 'pop')]
    L, x0, k, b = float(ret['L'].values[0]), float(ret['x0'].values[0]), float(ret['k'].values[0]),  float(ret['b'].values[0])
    return sigmoid(t,L,x0,k,b)


#we perform exponential curve fitting on the cumax, cs and cl time series between 1961 and 2013. The parameters a and b are identified (see Methdos). These will be used later to project cumax, cl, cs to 2100 using the f (yield scneario) parameter.

dict_cl = {}
dict_cs = {}
dict_cumax = {}

for country in cc_iso3_map:
    reduced = cumaxclcs.loc[(cumaxclcs['country code'] == country)]
    cl_dict = dict(zip(reduced.year, reduced.cl))
    cs_dict = dict(zip(reduced.year, reduced.cs))
    cumax_dict = dict(zip(reduced.year, reduced.cumax))
    
    smaller =  param_res.loc[(param_res['country'] == country)]
    start,end = smaller['start'].values[0], smaller['end'].values[0]
    
    if start < 1990:
        t_ = range(1990,2014)
    else:
        t_ = range(start,end)
    
    x = np.array(list(t_))
    y_cl = np.array([cl_dict[y] for y in t_])
    y_cs = np.array([cs_dict[y] for y in t_])
    y_cumax = np.array([cumax_dict[y] for y in t_])

    b_cl, a_cl = np.polyfit(x, np.log(y_cl), 1, w=np.sqrt(y_cl))
    b_cs, a_cs = np.polyfit(x, np.log(y_cs), 1, w=np.sqrt(y_cs))
    b_cumax, a_cumax = np.polyfit(x, np.log(y_cumax), 1, w=np.sqrt(y_cumax))
    dict_cl[country] = (a_cl,b_cl)
    dict_cs[country] = (a_cs,b_cs)
    dict_cumax[country] = (a_cumax,b_cumax)
    
def get_income_conspd(i,t):
    t = int(t)
    return conspd.loc[(conspd['country code'] == i)&(conspd['year'] == t)]['income value'].values[0]

def get_cumax_proj(i,t):
    #returns the projected value of cumax of country i at year t
    if t < 2014:
        t = int(t)
        return cumaxclcs.loc[(cumaxclcs['country code'] == i)&(cumaxclcs['year'] == t)]['cumax'].values[0]
    else:
        a = dict_cumax[i][0]
        b = dict_cumax[i][1]
        y0 = get_cumax_proj(i,2013)
        #x0 = 2013
        if (f < 1) and (f > 0):
            if b<0:
                c = (-b)*exp_f(2013,a,b)/(y0 - y0*f)
                return y0 - (y0 - y0*f)*(1 - math.exp(-c*(t-2013)))
            else:
                c = (b)*exp_f(2013,a,b)/(y0*f)
                return y0 + (y0*f)*(1 - math.exp(-c*(t-2013)))
        elif(f == 0):
            return exp_f(t,a,b)
        else:
            return y0 


def get_cs_proj(i,t):
    #returns the projected value of cs of country i at year t
    
    if t < 2014:
        t = int(t)
        return cumaxclcs.loc[(cumaxclcs['country code'] == i)&(cumaxclcs['year'] == t)]['cs'].values[0]
    else:
        a = dict_cs[i][0]
        b = dict_cs[i][1]
        y0 = get_cs_proj(i,2013)
        #x0 = 2013
        if (f < 1) and (f > 0):
            if b<0:
                c = (-b)*exp_f(2013,a,b)/(y0 - y0*f)
                return y0 - (y0 - y0*f)*(1 - math.exp(-c*(t-2013)))
            else:
                c = (b)*exp_f(2013,a,b)/(y0*f)
                return y0 + (y0*f)*(1 - math.exp(-c*(t-2013)))
        elif(f == 0):
            return exp_f(t,a,b)
        else:
            return y0 

    
def get_cl_proj(i,t):
    #returns the projected value of cl of country i at year t
    
    if t < 2014:
        t = int(t)
        return cumaxclcs.loc[(cumaxclcs['country code'] == i)&(cumaxclcs['year'] == t)]['cl'].values[0]
    else:
        a = dict_cl[i][0]
        b = dict_cl[i][1]
        y0 = get_cl_proj(i,2013)

        if (f < 1) and (f > 0) :
            if b<0:
                c = (-b)*exp_f(2013,a,b)/(y0 - y0*f)
                return y0 - (y0 - y0*f)*(1 - math.exp(-c*(t-2013)))
            else:
                c = (b)*exp_f(2013,a,b)/(y0*f)
                return y0 + (y0*f)*(1 - math.exp(-c*(t-2013)))
        elif(f == 0):
            return exp_f(t,a,b)
        else:
            return y0


def get_pop_ipcc(i,t):
    #returns the projected value of population of country i at year t according to SSP scenario
    
    if t <= 2013:
        t = int(t)
        return get_pop_proj(i,t)
    else:
        return float(dict_spline_pop[i](t))*(10**6)    

def get_income_ipcc(i,t):
    #returns the projected value of per capita gdp (PPP), income, of country i at year t according to SSP scenario
    
    if t <= 2013:
        t = int(t)
        return get_income_conspd(i,t)
    else:
        val = float(dict_spline_gdp[i](t))/float(dict_spline_pop[i](t))*(10**3)    
        return val/ratio_05_18


#normalizing all the income projections for countries included in the analysis with the largest value in the projected time-series

max_income = {}

for i in list(param_res['country']):
    start = param_res.loc[param_res['country'] == i]['start'].values[0]
    end = param_res.loc[param_res['country'] == i]['end'].values[0]
    max_ = np.nanmax([get_income_conspd(i,t_) for t_ in range(start,end+1)])
    max_income[i] = max_
                      
    
   
def get_income_ipcc_normalized(i,t):
    #returns the normalized ipcc income projection under SSP scenario for country i at year t
    m = get_income_ipcc(i,t)
    if m < max_income[i]:
        return m/max_income[i]
    else:
        return 1
    
   
def get_pov_proj(i,t):
    #returns the poverty projection for country i at year t (for details see Methods)
    
    pov = get_poverty(i,t)
    if pov<0:
        return 0
    elif pov>1:
        return 1
    else:
        return pov
    

def L_proj(i,x,t):
    #returns the global land use projection for consumption by country i at year t (for details see Methods)
    
    if t<=2013:
        return lu_calc.loc[(lu_calc['country code'] == i)&(lu_calc['year'] == int(t))]['land data'].values[0]
    else:
        cs = get_cs_proj(i,t)
        cl = get_cl_proj(i,t)
        pov = get_pov_proj(i,t)
        pop = get_pop_ipcc(i,t)
        if x>0.99999999:
            log_term = 0.05
        else:
            log_termD = -(math.log(1-x)) 
            log_term = 1/log_termD if log_termD!=0.0 else 100
            cumax_ = get_cumax_proj(i,t)
            
            if cl*log_term > cumax_:
                return (cumax_*(1 - pov) + pov*cs)*pop

        return (cl*log_term*(1 - pov) + pov*cs)*pop

    

#functions for getting parameters (deviated by desired percentages; 0 for baseline)

def get_kappa(i):
    val = param_res.loc[param_res['country'] == i]['kappa'].drop_duplicates().values[0]
    if val > 0:
        return (val)*((100+pct_kappa)/100)
    else:
        return (val)*((100-pct_kappa)/100)

def get_alpha(i):
    val = param_res.loc[param_res['country'] == i]['alpha'].drop_duplicates().values[0]
    if val > 0:
        return (val)*((100+pct_alpha)/100)
    else:
        return (val)*((100-pct_alpha)/100) 

def get_beta(i):
    val = param_res.loc[param_res['country'] == i]['beta'].drop_duplicates().values[0]
    if val > 0:
        return (val)*((100+pct_beta)/100)
    else:
        return (val)*((100-pct_beta)/100)


def coupled_projection(x_vect,t,i_dict):
    #defines the global coupled dynamical equation projecting the x for all countries included in the analysis
    
    max_land = 3645857954.9816337
    N = len(i_dict)
    x_prime_vect = np.zeros(N)
    L_vect = np.zeros(N)
    for j in i_dict:
        i = i_dict[j]
        L_vect[j] = L_proj(i,x_vect[j],t)
    for j in i_dict:
        i = i_dict[j]
        m = get_income_ipcc_normalized(i,t)
        x_prime_vect[j] = get_kappa(i)*x_vect[j]*(1 - x_vect[j])*(sum(L_vect)/max_land + get_alpha(i) + get_beta(i)*m)
    return x_prime_vect



#INITIALIZING the dynamical equation with x0 at 2011 with data
    
proj_start, proj_end = 2011, 2100
t_proj = np.linspace(proj_start,proj_end,proj_end - proj_start + 1)

x0_vect = []
for i in cc_iso3_map:

    x0_vect.append(lu_calc.loc[(lu_calc['country code'] == i)&(lu_calc['year'] == int(proj_start))]['xi'].values[0])

x0_vect = np.array(x0_vect)

#INTEGRATING THE COUPLED DIFFERENTIAL EQUATION
csvData = [['country','country name','scenario','year','land','xi','f','pop','income']]


frame_dict = {}


sol = odeint(coupled_projection, x0_vect, t_proj, args=(dictionary_country,))


#with the projection for x for all countries till 2100, corresponding land-use is calculated and stored in the list of list - csvData
for j in dictionary_country:
    i = dictionary_country[j]
    cc_name = areapd.loc[areapd['Country Code'] == i]['Country'].drop_duplicates().values[0]
    for k in range(0,len(t_proj)):
        y = int(t_proj[k])
        val__ = L_proj(i,sol[k][j],y) 
        csvData.append([i,cc_name,scenario,y,val__,sol[k][j],f,get_pop_ipcc(i,y),get_income_ipcc(i,y)])
     
    

#FOR YEARWISE COUNTRYWISE PROJECTIONS EXTRACT THE DATAFRAME: df    
df = pd.DataFrame(csvData,columns = ['country','country name','scenario','year','land','xi','f','pop','income'])
#df.to_csv(results/projections.csv)


#FOR FINDING THE PEAK YEAR AND PEAK LAND DO THE FOLLOWING (or else comment out):
year_global_land = {y: sum(df.loc[(df['year'] == y)]['land']) for y in t_proj}
peak_land = max([sum(df.loc[(df['year'] == y)]['land']) for y in t_proj])/10**9
peak_year =  max(year_global_land.items(), key=operator.itemgetter(1))[0]
frame_dict[f]= {'kappa': [pct_kappa], 'beta': [pct_beta], 'alpha':[pct_alpha], 'peak land': [peak_land], 'f':[f], 'scenario':[scen], 'peak year': [peak_year]}
dataf = pd.DataFrame.from_dict(frame_dict[f])
dataf.to_csv('results/' + scen + '_f%d_'%(f*10) + '_%d_%d_%d'%(pct_kappa,pct_alpha,pct_beta) +'.csv')
    




