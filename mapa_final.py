import plotly.express as px
import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
nazvanie_kolonki=['mfx_female','mfx_male']

df = pd.read_csv("D:\staff\sample_wvs.csv")

print(df)

df1 = pd.read_csv("D:\staff\iso.csv")
print(df1)

#subsample = df[(df['nomer'] == 1)&(df['x1'] == 1)]
#print(subsample)
for j in range (0,2):
      for i in range(1, 96):
            print('Nomer = ',i)
            subsample = df[(df['newid'] == i)&(df['x1'] == j)]
            model = smf.logit(formula='y ~ hedu + x2 +x2_2', data=subsample).fit()
            print(model.summary())
            #print(model.get_margeff(at ='mean').summary()) # get marginal effects
            margeff = model.get_margeff(at ='mean')
            #print(margeff.summary_frame())
            #l=margeff.summary_frame()
            #print(l)
            marginal_effect = margeff.summary_frame().iloc[0,0]*100
            pvalue_edu = margeff.summary_frame().iloc[0,3]
            if pvalue_edu>0.1:
                  marginal_effect =0
            if marginal_effect<0:
                  marginal_effect =0
            df1.at[i-1,nazvanie_kolonki[j]]= marginal_effect

print(df1)
fig1 = px.scatter_geo(df1, locations="iso_alpha", hover_name="country", size="mfx_male",
                     projection="natural earth", title="Figure 1. Marginal effect of education : Males")
fig1.show()

fig2 = px.scatter_geo(df1, locations="iso_alpha", hover_name="country", size="mfx_female",
                     projection="natural earth",title="Figure 2. Marginal effect of education : Females")
fig2.show()
