import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
data=pd.read_csv('WorldHappiness_Corruption_2015_2020.csv')
print(data.columns)
print(data.happiness_score.describe())
print(data.freedom.info())
europe=data[(data.continent)=='Europe']
print(europe)
asia=data[(data.continent)=='Asia']
print(asia)
africa=data[(data.continent)=='Africa']
print(africa)
n_america=data[(data.continent)=="North America"]
print(n_america)
s_america=data[(data.continent)=="South America"]
print(s_america)
australia=data[(data.continent)=="Australia"]
print(australia)
print(data.continent.value_counts())
#IN Asia
asi=asia.Country.values[20:30]
asi_happiness_score=asia.happiness_score.values[20:30]
sn.lineplot(asi,asi_happiness_score)
plt.show()
asi_free=asia.freedom.values[20:30]
asi_gov=asia.government_trust.values[20:30]
plt.plot(asi_free)
plt.plot(asi)
plt.ylabel('names of countries in asia')
plt.legend()
plt.show()
year=asia.Year.values[0:100]
gdp=asia.gdp_per_capita.values[0:100]
plt.scatter(year,gdp)
plt.xlabel('year')
plt.ylabel('gdp')
plt.show()
year=np.array(asia.Year.values[0:75])
yr=year.reshape(-1,1)
gdp=np.array(asia.gdp_per_capita.values[0:75])
gp=gdp.reshape(-1,1)
year_pre=np.array(asia.Year.values[75:200])
year_pr=year_pre.reshape(-1,1)
gd=np.array(asia.gdp_per_capita.values[75:200])
gpd=gd.reshape(-1,1)
from sklearn.linear_model import LinearRegression
linear=LinearRegression()
linear.fit(yr,gp)
pred=linear.predict(year_pr)
print(pred)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(pred,gpd)
print('The mse value is asia',mse)
print('the RMSE value in asia ',np.sqrt(mse))
plt.scatter(pred,gpd)
plt.xlabel('predicted vs actual one')
plt.legend()
plt.show()
happiness_score=asia.happiness_score.values[0:100]
plt.plot(year)
plt.xlabel('year')
plt.plot(happiness_score)
plt.ylabel('happiness_Score')
plt.legend()
plt.show()
#same procedure follows for all continents