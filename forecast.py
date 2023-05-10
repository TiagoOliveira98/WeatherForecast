###### Import Dependencies ######

# Library to load and treat data
import pandas as pd
#Library for framework for interpretable time series forecasting
from neuralprophet import NeuralProphet
#Library for plotting
from matplotlib import pyplot as plt
#Library to save the model
import pickle

###### Read Data ######
df = pd.read_csv('./data/weatherAUS.csv')
print(df.head())
#Verify the locations existing in the data
print(df.Location.unique())
#Let's choose Melbourne!
#Verify the features available in data
print(df.columns)
#Let´s forecast the temperature at 3pm!

#Check features type
print(df.dtypes)
#Date is of object type, it is necessary to convert it into time
#We are only going to utilize the Melbourne data we can trim everything else
melbdf = df[df['Location']=='Melbourne']
melbdf['Date'] = pd.to_datetime(melbdf['Date'])
print(melbdf.head())
print(melbdf.dtypes)

#PLot the temperature overtime
plt.plot(melbdf['Date'],melbdf['Temp3pm'])
plt.show()
#We can see a large amount of data that is missing

#Let's treat the data and remove this gap (Cut everything after 2015)
#Create a column with teh year feature
melbdf['Year'] = melbdf['Date'].apply(lambda x: x.year)
melbdf = melbdf[melbdf['Year']<=2015]
plt.plot(melbdf['Date'],melbdf['Temp3pm'])
plt.show()

#We need trim the data since the framework expects only two columns: ds, y
data = melbdf[['Date','Temp3pm']]
#Drop any NA values
data.dropna(inplace=True)
data.columns = ['ds', 'y']
print(data.head())

###### Train Model ######
model = NeuralProphet()
#Uses ARNet in teh background that is a good neural network in time series forecasting
model.fit(data, freq='D', epochs=500)
# 1000 epochs -> loss=0.0116, v_num=0, MAE=3.020, RMSE=3.920, Loss=0.0108, RegLoss=0.000
# 500 epochs -> loss=0.0102, v_num=1, MAE=3.020, RMSE=3.930, Loss=0.0108, RegLoss=0.000
# 2000 epochs -> loss=0.0105, v_num=2, MAE=3.010, RMSE=3.920, Loss=0.0108, RegLoss=0.000

###### Forecast ######
#Let's make a dataframe with enlarged time
future = model.make_future_dataframe(data, periods=900)
forecast = model.predict(future)
print(forecast.head())
#Plot the forecast
plot1 = model.plot(forecast)

###### Save Model ######
with open('forecast_model.pkl', 'wb') as f:
    pickle.dump(model,f)

#To load
#with open('forecast_model.pkl', 'rb') as f:
#    model = pickle.load(f)###### Import Dependencies ######

# Library to load and treat data
import pandas as pd
#Library for framework for interpretable time series forecasting
from neuralprophet import NeuralProphet
#Library for plotting
from matplotlib import pyplot as plt
#Library to save the model
import pickle

###### Read Data ######
df = pd.read_csv('./data/weatherAUS.csv')
print(df.head())
#Verify the locations existing in the data
print(df.Location.unique())
#Let's choose Melbourne!
#Verify the features available in data
print(df.columns)
#Let´s forecast the temperature at 3pm!

#Check features type
print(df.dtypes)
#Date is of object type, it is necessary to convert it into time
#We are only going to utilize the Melbourne data we can trim everything else
melbdf = df[df['Location']=='Melbourne']
melbdf['Date'] = pd.to_datetime(melbdf['Date'])
print(melbdf.head())
print(melbdf.dtypes)

#PLot the temperature overtime
plt.plot(melbdf['Date'],melbdf['Temp3pm'])
plt.show()
#We can see a large amount of data that is missing

#Let's treat the data and remove this gap (Cut everything after 2015)
#Create a column with teh year feature
melbdf['Year'] = melbdf['Date'].apply(lambda x: x.year)
melbdf = melbdf[melbdf['Year']<=2015]
plt.plot(melbdf['Date'],melbdf['Temp3pm'])
plt.show()

#We need trim the data since the framework expects only two columns: ds, y
data = melbdf[['Date','Temp3pm']]
#Drop any NA values
data.dropna(inplace=True)
data.columns = ['ds', 'y']
print(data.head())

###### Train Model ######
model = NeuralProphet()
#Uses ARNet in teh background that is a good neural network in time series forecasting
model.fit(data, freq='D', epochs=500)
# 1000 epochs -> loss=0.0116, v_num=0, MAE=3.020, RMSE=3.920, Loss=0.0108, RegLoss=0.000
# 500 epochs -> loss=0.0102, v_num=1, MAE=3.020, RMSE=3.930, Loss=0.0108, RegLoss=0.000
# 2000 epochs -> loss=0.0105, v_num=2, MAE=3.010, RMSE=3.920, Loss=0.0108, RegLoss=0.000

###### Forecast ######
#Let's make a dataframe with enlarged time
future = model.make_future_dataframe(data, periods=900)
forecast = model.predict(future)
print(forecast.head())
#Plot the forecast
plot1 = model.plot(forecast)

###### Save Model ######
with open('forecast_model.pkl', 'wb') as f:
    pickle.dump(model,f)

#To load
#with open('forecast_model.pkl', 'rb') as f:
#    model = pickle.load(f)