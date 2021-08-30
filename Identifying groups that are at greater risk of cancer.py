#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data wrangling tools
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# statistical analysis
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# datadotworld SDK
import datadotworld as ddw
get_ipython().run_line_magic('load_ext', 'watermark')

import time


# In[ ]:


get_ipython().run_line_magic('watermark', '-v -p pandas,numpy,matplotlib,datadotworld')


# # 1. Acquire and Prepare Data    
# ### Cancer Incidence & Mortality Data
# Cancer mortality and incidence data can be found at cancer.gov

# In[ ]:


# retrieve the cancer data from data.world
mortdf = ddw.query('nrippner/cancer-analysis-hackathon-challenge',
                  'SELECT * FROM `death .csv/death `').dataframe

incddf = ddw.query('nrippner/cancer-analysis-hackathon-challenge',
                        'SELECT * FROM `incd.csv/incd`').dataframe

mortdf = mortdf[mortdf.FIPS.notnull()]
incddf = incddf[incddf.FIPS.notnull()]

mortdf['FIPS'] = mortdf.FIPS.apply(lambda x: str(int(x)))                            .astype(np.object_)                            .str.pad(5, 'left', '0')

incddf['FIPS'] = incddf.FIPS.apply(lambda x: str(int(x)))                            .astype(np.object_)                            .str.pad(5, 'left', '0')

incddf.drop(incddf.columns[[0,3,4,7,8,9]].values, axis=1, inplace=True)
mortdf.drop(mortdf.columns[[0,2,4,5,7,8,9,10]], axis=1, inplace=True)

incddf.rename(columns={incddf.columns[1]:'Incidence_Rate',
                       incddf.columns[2]:'Avg_Ann_Incidence'}, inplace=True)
mortdf.rename(columns={mortdf.columns[1]:'Mortality_Rate',
                       mortdf.columns[2]:'Avg_Ann_Deaths'}, inplace=True)


# #### A. Poverty Data

# B17001_002
# POVERTY STATUS IN THE PAST 12 MONTHS BY SEX BY AGE for Population For Whom Poverty Status Is Determined% Income in the past 12 months below poverty level    
# B17001_003
# POVERTY STATUS IN THE PAST 12 MONTHS BY SEX BY AGE for Population For Whom Poverty Status Is Determined% Income in the past 12 months below poverty level:% Male    
# B17001_017
# POVERTY STATUS IN THE PAST 12 MONTHS BY SEX BY AGE for Population For Whom Poverty Status Is Determined% Income in the past 12 months below poverty level:% Female

# In[ ]:


# Retrieve a list of table names (by state)
pov = ddw.load_dataset('uscensusbureau/acs-2015-5-e-poverty')

tables = []
for i in pov.tables:
    if len(i) == 2:
        tables.append(i)

# remove Puerto Rico
tables.remove('pr')


# In[ ]:


print(len(tables))
np.array(tables)


# In[ ]:


# Retrieve the Census poverty data from data.world
start = time.time()

# a string - the poverty columns we want from the Census ACS 
cols = '`State`, `StateFIPS`, `CountyFIPS`, `AreaName`, `B17001_002`, `B17001_003`,'       '`B17001_017`'

# call the data for each state and concatenate
for i, state in enumerate(tables):
    if i == 0:
        povdf = ddw.query('uscensusbureau/acs-2015-5-e-poverty',
                  '''SELECT %s FROM `AK`
                     WHERE SummaryLevel=50''' % cols).dataframe 
    else:
        df = ddw.query('uscensusbureau/acs-2015-5-e-poverty',
                       '''SELECT %s FROM `%s`
                          WHERE SummaryLevel=50''' % (cols, state.upper())).dataframe
    
        povdf = pd.concat([povdf, df], ignore_index=True)

end = time.time()

print(end - start)


# Add leading zeros to the state and county FIPS codes
povdf['StateFIPS'] = povdf.StateFIPS.astype(np.object_)                                    .apply(lambda x: str(x))                                    .str.pad(2, 'left', '0')
povdf['CountyFIPS'] = povdf.CountyFIPS.astype(np.object_)                                      .apply(lambda x: str(x))                                      .str.pad(3, 'left', '0')

povdf.rename(columns={'B17001_002':'All_Poverty', 'B17001_003':'M_Poverty', 'B17001_017':'F_Poverty'},
             inplace=True)


# In[ ]:


povdf.head()


# #### B. Income Data

# B19013_001
# MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2015 INFLATION-ADJUSTED DOLLARS) for Households%Median household income in the past 12 months (in 2015 Inflation-adjusted dollars)                
# B19013A_001
# MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2015 INFLATION-ADJUSTED DOLLARS) (WHITE ALONE HOUSEHOLDER) for Households With A Householder Who Is White Alone%Median household income in the past 12 months (in 2015 Inflation-adjusted dollars)       
#        
# B19013B_001
# MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2015 INFLATION-ADJUSTED DOLLARS) (BLACK OR AFRICAN AMERICAN ALONE HOUSEHOLDER) for Households With A Householder Who Is Black Or African American Alone%Median household income in the past 12 months (in 2015 Inflation-adjusted dollars)       
#        
# B19013C_001
# MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2015 INFLATION-ADJUSTED DOLLARS) (AMERICAN INDIAN AND ALASKA NATIVE ALONE HOUSEHOLDER) for Households With A Householder Who Is American Indian And Alaska Native Alone%Median household income in the past 12 months (in 2015 Inflation-adjusted dollars)       
#             
# B19013D_001
# MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2015 INFLATION-ADJUSTED DOLLARS) (ASIAN ALONE HOUSEHOLDER) for Households With A Householder Who Is Asian Alone%Median household income in the past 12 months (in 2015 Inflation-adjusted dollars)      
#          
#        
# B19013I_001
# MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2015 INFLATION-ADJUSTED DOLLARS) (HISPANIC OR LATINO HOUSEHOLDER) for Households With A Householder Who Is Hispanic Or Latino%Median household income in the past 12 months (in 2015 Inflation-adjusted dollars)

# In[ ]:


cols = '`StateFIPS`, `CountyFIPS`,'       '`B19013_001`, `B19013A_001`, `B19013B_001`, `B19013C_001`, `B19013D_001`,'       '`B19013I_001`'

start = time.time()

for i, state in enumerate(tables):
    if i == 0:
        incomedf = ddw.query('uscensusbureau/acs-2015-5-e-income',
                  '''SELECT %s FROM `AK`
                     WHERE SummaryLevel=50''' % cols).dataframe 
    else:
        df = ddw.query('uscensusbureau/acs-2015-5-e-income',
                       '''SELECT %s FROM `%s`
                          WHERE SummaryLevel=50''' % (cols, state.upper())).dataframe
        incomedf = pd.concat([incomedf, df], ignore_index=True)

        end = time.time()

print(end - start)

incomedf['StateFIPS'] = incomedf.StateFIPS.astype(np.object_)                                .apply(lambda x: str(x))                                .str.pad(2, 'left', '0')
incomedf['CountyFIPS'] = incomedf.CountyFIPS.astype(np.object_)                                 .apply(lambda x: str(x))                                 .str.pad(3, 'left', '0')

incomedf.rename(columns={'B19013_001':'Med_Income', 'B19013A_001':'Med_Income_White', 
                         'B19013B_001':'Med_Income_Black', 'B19013C_001':'Med_Income_Nat_Am',
                         'B19013D_001':'Med_Income_Asian', 'B19013I_001':'Hispanic'}, inplace=True)


# #### C. Health Insurance Data

# B27001_004
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% Under 6 years            
# B27001_005
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% Under 6 years:% With health insurance coverage       
# B27001_007
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 6 to 17 years:% With health insurance coverage            
# B27001_008
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 6 to 17 years:% No health insurance coverage               
# B27001_010
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 18 to 24 years:% With health insurance coverage          
# B27001_011
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 18 to 24 years:% No health insurance coverage            
# B27001_013
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 25 to 34 years:% With health insurance coverage             
# B27001_014
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 25 to 34 years:% No health insurance coverage             
# B27001_016
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 35 to 44 years:% With health insurance coverage          
# B27001_017
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 35 to 44 years:% No health insurance coverage              
# B27001_019
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 45 to 54 years:% With health insurance coverage        
# B27001_020
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 45 to 54 years:% No health insurance coverage          
# B27001_022
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 55 to 64 years:% With health insurance coverage            
# B27001_023
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 55 to 64 years:% No health insurance coverage           
# B27001_025
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 65 to 74 years:% With health insurance coverage              
# B27001_026
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 65 to 74 years:% No health insurance coverage             
# B27001_028
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 75 years and over:% With health insurance coverage            
# B27001_029
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Male:% 75 years and over:% No health insurance coverage               
# B27001_032
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% Under 6 years:                 
# B27001_033
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% Under 6 years:% With health insurance coverage       
# B27001_035
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 6 to 17 years:% With health insurance coverage         
# B27001_036
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 6 to 17 years:% No health insurance coverage            
# B27001_038
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 18 to 24 years:% With health insurance coverage               
# B27001_039
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 18 to 24 years:% No health insurance coverage              
# B27001_041
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 25 to 34 years:% With health insurance coverage            
# B27001_042
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 25 to 34 years:% No health insurance coverage          
# B27001_044
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 35 to 44 years:% With health insurance coverage          
# B27001_045
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 35 to 44 years:% No health insurance coverage          
# B27001_047
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 45 to 54 years:% With health insurance coverage             
# B27001_048
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 45 to 54 years:% No health insurance coverage           
# B27001_050
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 55 to 64 years:% With health insurance coverage          
# B27001_051
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 55 to 64 years:% No health insurance coverage           
# B27001_053
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 65 to 74 years:% With health insurance coverage            
# B27001_054
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 65 to 74 years:% No health insurance coverage              
# B27001_056
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 75 years and over:% With health insurance coverage          
# B27001_057
# HEALTH INSURANCE COVERAGE STATUS BY SEX BY AGE for Civilian Noninstitutionalized Population% Female:% 75 years and over:% No health insurance coverage           

# In[ ]:


cols = '`StateFIPS`, `CountyFIPS`,'       '`B27001_004`, `B27001_005`, `B27001_007`, `B27001_008`,'       '`B27001_010`, `B27001_011`, `B27001_013`, `B27001_014`,'       '`B27001_016`, `B27001_017`, `B27001_019`, `B27001_020`,'       '`B27001_022`, `B27001_023`, `B27001_025`, `B27001_026`,'       '`B27001_028`, `B27001_029`, `B27001_032`, `B27001_033`,'       '`B27001_035`, `B27001_036`, `B27001_038`, `B27001_039`,'       '`B27001_041`, `B27001_042`, `B27001_044`, `B27001_045`,'       '`B27001_047`, `B27001_048`, `B27001_050`, `B27001_051`,'       '`B27001_053`, `B27001_054`, `B27001_056`, `B27001_057`'
# male <= 029   

start = time.time()

for i, state in enumerate(tables):
    if i == 0:
        hinsdf = ddw.query('uscensusbureau/acs-2015-5-e-healthinsurance',
                  '''SELECT %s FROM `AK`
                     WHERE SummaryLevel=50''' % cols).dataframe 
   
    else:
        df = ddw.query('uscensusbureau/acs-2015-5-e-healthinsurance',
                       '''SELECT %s FROM `%s`
                          WHERE SummaryLevel=50''' % (cols, state.upper())).dataframe
        hinsdf = pd.concat([hinsdf, df], ignore_index=True)

end = time.time()
print(end - start)

hinsdf['StateFIPS'] = hinsdf.StateFIPS.astype(np.object_)                                      .apply(lambda x: str(x))                                      .str.pad(2, 'left', '0')
hinsdf['CountyFIPS'] = hinsdf.CountyFIPS.astype(np.object_)                                        .apply(lambda x: str(x))                                        .str.pad(3, 'left', '0')


# In[ ]:


# columns representing males' health insurance statistics
males = ['`B27001_004`', '`B27001_005`', '`B27001_007`', '`B27001_008`',
           '`B27001_010`', '`B27001_011`', '`B27001_013`', '`B27001_014`',
           '`B27001_016`', '`B27001_017`', '`B27001_019`', '`B27001_020`',
           '`B27001_022`', '`B27001_023`', '`B27001_025`', '`B27001_026`',
           '`B27001_028`', '`B27001_029`']

# females' health insurance statistics
females = ['`B27001_032`', '`B27001_033`', '`B27001_035`', '`B27001_036`', 
           '`B27001_038`', '`B27001_039`', '`B27001_041`', '`B27001_042`', 
           '`B27001_044`', '`B27001_045`', '`B27001_047`', '`B27001_048`', 
           '`B27001_050`', '`B27001_051`', '`B27001_053`', '`B27001_054`', 
           '`B27001_056`', '`B27001_057`']

# separate the "with" and "without" health insurance columns
males_with = []
males_without = []
females_with = []
females_without = []

# strip the backticks
for i, j in enumerate(males):
    if i % 2 == 0:
        males_with.append(j.replace('`', ''))
    else:
        males_without.append(j.replace('`', ''))
        
for i, j in enumerate(females):
    if i % 2 == 0:
        females_with.append(j.replace('`', ''))
    else:
        females_without.append(j.replace('`', ''))

# Create features that sum all the individual age group
clist = [males_with, males_without, females_with, females_without]
newcols = ['M_With', 'M_Without', 'F_With', 'F_Without'] 

for col in newcols:
    hinsdf[col] = 0

for i in males_with:
    hinsdf['M_With'] += hinsdf[i]  
for i in males_without:
    hinsdf['M_Without'] += hinsdf[i]
for i in females_with:
    hinsdf['F_With'] += hinsdf[i]
for i in females_without:
    hinsdf['F_Without'] += hinsdf[i]

hinsdf['All_With'] = hinsdf.M_With + hinsdf.F_With
hinsdf['All_Without'] = hinsdf.M_Without + hinsdf.F_Without

# Remove all the individual age group variables
# but, save them as a df called hinsdf_extra (just in case)
hinsdf_extra = df.loc[:, df.columns[df.columns.str.contains('B27001')].values]
hinsdf.drop(df.columns[df.columns.str.contains('B27001')].values, axis=1, inplace=True)


# ### Merge DataFrames    

# In[ ]:


dfs = [povdf, incomedf, hinsdf, incddf, mortdf]


# In[ ]:


# create FIPS features
for df in [povdf, incomedf, hinsdf]:
    df['FIPS'] = df.StateFIPS + df.CountyFIPS
    df.drop(['StateFIPS', 'CountyFIPS'], axis=1, inplace=True)


# In[ ]:


# use the pandas isin() method to see the intersections of FIPS features
# across dataframes
[[i, ii, sum(pd.Series(j.FIPS.unique()).isin(dfs[ii].FIPS))] 
    for i, j in enumerate(dfs) for ii in range(len(dfs))]


# In[ ]:


# look at the number of unique FIPS values per dataframe
for i in dfs:
    print(len(i.FIPS.unique()))


# In[ ]:


# check to see if all the FIPS values are 5 digits in length
dfs = [povdf, incomedf, hinsdf, incddf, mortdf] # our 5 dataframes to merge
for i, j in enumerate(dfs):
    lens = []
    for fips in j.FIPS.values:
        lens.append(len(fips))
    print(pd.Series(lens).value_counts(), '\n', '-'*10)    


# In[ ]:


for i, j in enumerate(dfs):
    if i == 0:
        fulldf = j.copy()
    else:
        fulldf = fulldf.merge(j, how='inner', on='FIPS')


# # 2. Exploratory Analysis (and continued data cleaning)

# In[ ]:


fulldf.shape


# In[ ]:


fulldf.head()


# In[ ]:


# check for null values
for col in fulldf.columns:
    print((col, sum(fulldf[col].isnull())))


# In[ ]:


fulldf.drop(['Med_Income_White', 'Med_Income_Black', 'Med_Income_Nat_Am',
             'Med_Income_Asian', 'Hispanic'], axis=1, inplace=True)


# In[ ]:


data_dict = pd.DataFrame(fulldf.columns.values, index=range(len(fulldf.columns)), columns=['Feature'])

data_dict['Definition'] = ['','','Both male and female reported below poverty line (Raw)', 
                           'Males below poverty (Raw)', 'Females below poverty (Raw)', 'State + County FIPS (Raw)',
                           'Med_Income all enthnicities (Raw)', 'Males with health insurance (Raw)',
                           'Males without health insurance (Raw)', 'Females with health insurance (Raw)',
                           'Females without health insurance (Raw)', 'Males and Femaes with health ins. (Raw)',
                           'Males an Females without health ins (Raw)', 'Lung cancer incidence rate (per 100,000)',
                           'Average lung cancer incidence rate (Raw)', 'Recent trend (incidence)', 
                           'Lung cancer mortality rate (per 100,000)', 'Average lung cancer mortalities (Raw)']

data_dict['Notes'] = ''
data_dict.loc[[13,16], 'Notes'] = "'*' = fewer that 16 reported cases"

data_dict


# In[ ]:


def get_types(col_name):
    ts = (pd.Series([type(i) for i in fulldf[col_name]]).value_counts())
    print("%s\n" % feature, ts, "\n", "-"*30)

for feature in fulldf.columns:
    get_types(feature)


# ### Columns that seem to need to be fixed: Med_Income, Incidence_Rate, Avg_Ann_Incidence, Mortality_Rate, Avg_Ann_Deaths

# In[ ]:


# Mortality_Rate
# This script isolates values that fail to convert to numeric
def f(column):
    types = []
    for _, j in enumerate(column):
        try:
            pd.to_numeric(j)
            
        except:
            types.append(j)
    print(pd.Series(types).value_counts())

f(fulldf.Mortality_Rate)


# In[ ]:


# which states are associated with the "*"s?
fulldf.loc[fulldf.Incidence_Rate=='*', 'State'].value_counts()


# In[ ]:


# Histogram to see how exceptional a low rate of mortality is
mhist = pd.to_numeric(fulldf.Mortality_Rate[fulldf.Mortality_Rate != '*'])
print("min", mhist.min(), "max", mhist.max())
mhist.hist(figsize={14,7}, bins=20);


# In[ ]:


populationdf = ddw.query('nrippner/us-population-estimates-2015',
                         '''SELECT `POPESTIMATE2015`, `STATE`, `COUNTY`
                            FROM `CO-EST2015-alldata`''').dataframe


# In[ ]:


populationdf.shape


# In[ ]:


populationdf.head()


# In[ ]:


state = populationdf.STATE.apply(lambda x: str(x))                          .str.pad(2, 'left', '0')
county = populationdf.COUNTY.apply(lambda x: str(x))                            .str.pad(3, 'left', '0')

populationdf['FIPS'] = state + county

populationdf.head()


# In[ ]:


# first, let's check to see that the FIPS columns match up properly
print(sum(pd.Series(populationdf.FIPS.unique()).isin(fulldf.FIPS)), 'matches out of')
print("%d unique values" % len(populationdf.FIPS.unique()))


# In[ ]:


print(sum(pd.Series(fulldf.FIPS.unique()).isin(populationdf.FIPS)), 'matches out of')
print("%d unique values" % len(fulldf.FIPS.unique()))


# In[ ]:


fulldf = fulldf.merge(populationdf[['FIPS', 'POPESTIMATE2015']], on='FIPS', how='inner')


# In[ ]:


# Find median as reference point
fulldf.POPESTIMATE2015.median()
print("median, mean population:         %.1f, %.1f" % (fulldf.POPESTIMATE2015.median(),
                                         fulldf.POPESTIMATE2015.mean()))
print("median, mean population '*':      %.1f,   %.1f" % (fulldf.POPESTIMATE2015[fulldf.Mortality_Rate == '*'].median(),
                                            fulldf.POPESTIMATE2015[fulldf.Mortality_Rate == '*'].mean()))
print("median, mean population not '*': %.1f, %.1f" % (fulldf.POPESTIMATE2015[fulldf.Mortality_Rate != '*'].median(),
                                                fulldf.POPESTIMATE2015[fulldf.Mortality_Rate != '*'].mean()))


# In[ ]:


fulldf.POPESTIMATE2015[fulldf.Mortality_Rate == '*'].hist(figsize=(8,3), bins=30);


# In[ ]:


population_levels = [0, 1000, 5000, 10000, 15000, 20000, 50000, 100000, 500000, 10**6]
for i in range(1,len(population_levels)):
    print("population:", "%d-%d" % (population_levels[i-1], population_levels[i]),"median mort. rate:",
                                    fulldf.Mortality_Rate[(fulldf.Mortality_Rate != '*') &
                                                          (population_levels[i-1] < fulldf.POPESTIMATE2015) &
                                                          (fulldf.POPESTIMATE2015 < population_levels[i])].median())


# In[ ]:


print("Not '*'")
for i in range(1, len(population_levels)):
    print("# records between","%d-%d population" % (population_levels[i-1], population_levels[i]), 
                                      fulldf[(fulldf.Mortality_Rate != '*') &
                                             (fulldf.POPESTIMATE2015 <= population_levels[i]) &
                                             (fulldf.POPESTIMATE2015 > population_levels[i-1])].shape[0]) 


# In[ ]:


# a closer look at number of records for very low-population counties (< 5000)
print("not '*'")
population_levels = [2500, 3000, 3500, 4000, 4500, 5000]

for i in range(1, len(population_levels)):
    print("# records between","%d-%d population" % (population_levels[i-1], population_levels[i]), 
                                      fulldf[(fulldf.Mortality_Rate != '*') &
                                             (fulldf.POPESTIMATE2015 <= population_levels[i]) &
                                             (fulldf.POPESTIMATE2015 > population_levels[i-1])].shape[0]) 

print("'*'") 
for i in range(1, len(population_levels)):
    print("# records between","%d-%d population" % (population_levels[i-1], population_levels[i]), 
                                      fulldf[(fulldf.Mortality_Rate == '*') &
                                             (fulldf.POPESTIMATE2015 <= population_levels[i]) &
                                             (fulldf.POPESTIMATE2015 > population_levels[i-1])].shape[0]) 


# In[ ]:


fulldf = fulldf[fulldf.Mortality_Rate != '*']


# ### Clean up Med_Income, Incidence_Rate, Avg_Ann_Incidence, and Avg_Ann_Deaths

# In[ ]:


# Med_Income
fulldf['Med_Income'] = pd.to_numeric(fulldf.Med_Income)  # That was easy!


# In[ ]:


# Incidence_Rate
# Let's use this script to see which values fail to convert to numeric:
values = []
for _, j in enumerate(fulldf.Incidence_Rate):
    try:
        pd.to_numeric(j)
    except:
        values.append(j)
        
pd.Series(values).value_counts()[:10]


# In[ ]:


# create dummy variables for "Recent Trend"

# rename 'Recent Trend' to remove the space
fulldf.rename(columns={'Recent Trend':'RecentTrend'}, inplace=True)

# change all the missing values to the mode ('stable')
fulldf.replace({'RecentTrend' : {'*':'stable'}}, inplace=True)

# function to do boolean check and return 1 or 0
def f(x, term):
    if x == term:
        return 1
    else:
        return 0

# create new features using the apply method with the 'f' function we defined above
fulldf['Rising'] = fulldf.RecentTrend.apply(lambda x: f(x, term='rising'))
fulldf['Falling'] = fulldf.RecentTrend.apply(lambda x: f(x, term='falling'))

# Note that of the 3 levels of RecentTrend, we only created dummies for rising and falling
# We will be incuding constant in our model (dummy variable trap)


# In[ ]:


fulldf['RecentTrend'].value_counts()


# In[ ]:


y = pd.to_numeric(fulldf.Mortality_Rate).values
X = fulldf.loc[:,['All_Poverty', 'M_Poverty', 'F_Poverty', 'Med_Income',
            'M_With', 'M_Without', 'F_With', 'F_Without', 'All_With',
            'All_Without', 'Incidence_Rate', 'Falling', 'Rising',
            'POPESTIMATE2015']]


# In[ ]:


X.head()


# In[ ]:


X['Incidence_Rate'] = pd.to_numeric(X.Incidence_Rate, errors='coerce')


# In[ ]:


X['Incidence_Rate'] = X.Incidence_Rate.fillna(X.Incidence_Rate.median())
print(sum(X.Incidence_Rate.isnull()))


# In[ ]:


for col in ['All_Poverty', 'M_Poverty', 'F_Poverty', 'M_With',
            'M_Without', 'F_With', 'F_Without', 'All_With', 'All_Without']:
       
    X[col + "_PC"] = X[col] / X.POPESTIMATE2015 * 10**5


# In[ ]:


X.head()


# ## Visual Exploratory Analysis

# In[ ]:


# scatterplots (hat tip Sebastian Raschka from his book "Python Machine Learning")
sns.set(style='whitegrid', context='notebook')
sns.pairplot(X[['All_Poverty_PC', 'M_Poverty_PC', 'F_Poverty_PC', 'Med_Income']], size=2)
plt.show()


# In[ ]:


cols = ['All_Poverty_PC', 'M_Poverty_PC', 'F_Poverty_PC', 'Med_Income']
cm = np.corrcoef(X[['All_Poverty_PC', 'M_Poverty_PC', 'F_Poverty_PC', 'Med_Income']].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':15},
                 yticklabels=cols, xticklabels=cols)
plt.show()


# In[ ]:


X.drop(['M_Poverty_PC', 'F_Poverty_PC'], axis=1, inplace=True)


# In[ ]:


X.drop(['M_With_PC', 'F_With_PC'], axis=1, inplace=True)
X.drop(['M_Without_PC', 'F_Without_PC'], axis=1, inplace=True)
X.head()


# In[ ]:


cols = ['All_Poverty_PC', 'Med_Income', 'All_With_PC', 'All_Without_PC',
                'Incidence_Rate', 'POPESTIMATE2015']
sns.set(style='whitegrid', context='notebook')
sns.pairplot(X[cols], size=2)
plt.show()


# In[ ]:


cm = np.corrcoef(X[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':15},
                 yticklabels=cols, xticklabels=cols)
plt.show()


# # 3. Linear Regression Model

# In[ ]:


cols = ['All_Poverty_PC', 'Med_Income', 'All_With_PC',  'All_Without_PC',
        'Incidence_Rate', 'POPESTIMATE2015', 'Falling', 'Rising', 'All_Poverty',
        'All_With', 'All_Without']


# In[ ]:


# add constant (coloumn vector of all 1s)
X = X[cols]
X['Constant'] = 1
X.reset_index(drop=True, inplace=True)


# In[ ]:


# Fit linear regression model
lr = sm.OLS(y, X, hasconst=True)
result = lr.fit()


# In[ ]:


result.summary()


# #### Multicollinearity     

# In[ ]:


pd.DataFrame([[var, variance_inflation_factor(X.values, X.columns.get_loc(var))] for var in X.columns],
                   index=range(X.shape[1]), columns=['Variable', 'VIF'])


# In[ ]:


X.columns


# In[ ]:


vcols = ['All_Poverty_PC', 'Med_Income', 'All_With_PC', 'All_Without_PC',
       'Incidence_Rate', 'Falling', 'Rising', 'Constant']


# In[ ]:


Xvcols = X[vcols].reset_index(drop=True)
pd.DataFrame([[var, variance_inflation_factor(Xvcols.values, Xvcols.columns.get_loc(var))] for var in vcols],
                   index=range(len(vcols)), columns=['Variable', 'VIF'])


# In[ ]:


vcols = ['All_Poverty_PC', 'Med_Income', 'All_Without_PC',
       'Incidence_Rate', 'Falling', 'POPESTIMATE2015', 'Constant']
Xvcols = X[vcols].reset_index(drop=True)
lr = sm.OLS(y, Xvcols, hasconst=True)
result = lr.fit()


# In[ ]:


result.summary()


# **Are residuals normally distributed?**

# In[ ]:


# histogram superimposed by normal curve
plt.figure(figsize=(10,6))
import scipy.stats as stats
mu = np.mean(result.resid)
sigma = np.std(result.resid)
pdf = stats.norm.pdf(sorted(result.resid), mu, sigma)
plt.hist(result.resid, bins=100, normed=True)
plt.plot(sorted(result.resid), pdf, color='r', linewidth=2)
plt.show()


# In[ ]:


# QQplot
fig, [ax1, ax2] = plt.subplots(1,2, figsize=(10,3))
sm.qqplot(result.resid, stats.t, fit=True, line='45', ax = ax1)
ax1.set_title("t distribution")
sm.qqplot(result.resid, stats.norm, fit=True, line='45', ax=ax2)
ax2.set_title("normal distribution")
plt.show()


# **Heteroscedasticity**   

# In[ ]:


# plot predicted vs actual
plt.figure(figsize=(14,7))
sns.regplot(y, result.fittedvalues, line_kws={'color':'r', 'alpha':0.3, 
                                              'linestyle':'--', 'linewidth':2}, 
            scatter_kws={'alpha':0.5})
plt.ylim(0,160)
plt.xlabel('Actual Values')
plt.ylabel('Fitted Values')
plt.show()
print("Pearson R: ", stats.pearsonr(result.fittedvalues, y))


# In[ ]:


# plot actual values versus residuals
from statsmodels.nonparametric.smoothers_lowess import lowess
ys = lowess(result.resid.values, y, frac=0.2)
ys = pd.DataFrame(ys, index=range(len(ys)), columns=['a', 'b'])
ys = ys.sort_values(by='a')

fig, ax = plt.subplots(figsize=(14,9))
plt.scatter(y, result.resid, alpha=0.5, s=25)
plt.axhline(y=0, color='r', linestyle="--", alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")

plt.plot(ys.a, ys.b, c='green', linewidth=2, label="Lowess")
plt.legend()
plt.show()
print("Pearson R:", stats.pearsonr(y, result.resid))


# In[ ]:


# plot actual values versus residuals
plt.figure(figsize=(14,7))
plt.scatter(y=result.resid, x=result.fittedvalues, alpha=0.5, s=22)
plt.axhline(y=0, color='r', linestyle="--", alpha=0.5)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.ylim(-65, 50)
plt.show()

