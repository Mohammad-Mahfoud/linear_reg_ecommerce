
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

df = pd.read_csv("Dataset/Ecommerce Customers")


print(df.info())

## EDA :
sns.lmplot(x="Length of Membership",y="Yearly Amount Spent",data=df,scatter_kws={'alpha':0.3})
plt.show()


## creating our two datasets X (dataset of features (independent features without the unique string columns))
X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]

## creating the target dataset which is the dependent feature ( yearly amount spent)
y = df["Yearly Amount Spent"]

# preparing for splitting the dataset : 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

## Training the model:
model = LinearRegression()

## fitting the data into the model: 
model.fit(X_train,y_train)

## to see the most impact feature on our X ( the most imp weight(w) or coefficent on our X):
cdf = pd.DataFrame(model.coef_,X.columns,columns=["Coef"])
print(cdf)

# doing predictions on our model : 
y_pred = model.predict(X_test)

## now we do scatter plot to see the y_pred(our predictions) and y_test(our actual y values) and see if they form a straight line

sns.scatterplot(x=y_pred,y=y_test)
plt.xlabel("predictions")
plt.title("Evaluation of our Linear Regression Model")
plt.show()


## using performance metrics, we use mse and mae since its regression model: 

mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print(f"Mean squared error {mse}")
print(f"Mean absolute  error {mae}")
print("RMSE",math.sqrt(mean_squared_error(y_test,y_pred)))


## calculating the residuals : the distance between or real y_testing data and our y-pred(predicitons)

residuals = y_test - y_pred

sns.displot(residuals,bins=20,kde=True)
plt.show()