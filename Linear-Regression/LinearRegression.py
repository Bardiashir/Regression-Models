import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

# The goal is to predict the profit of a restaurant, based on the number of habitants where the restaurant
# is located. The chain already has several restaurants in different cities. The goal is to model
# the relationship between the profit and the populations from the cities where they are located.



data = pandas.read_csv('RegressionData.csv', header=None,
                       names=['X', 'y']) 
X = data['X'].values.reshape(-1, 1) 
y = data['y'] 
plt.scatter(X, y) 

# Linear regression using least squares optimization
reg = linear_model.LinearRegression() 
reg.fit(X, y)  

# Plot the linear fit
fig = plt.figure()
y_pred = reg.predict(X) 
plt.scatter(X, y, c='b')  
plt.plot(X, y_pred, 'r')  
fig.canvas.draw()
plt.show()


print("The linear relationship between X and y was modeled according to the equation: y = b_0 + X*b_1, \
where the bias parameter b_0 is equal to ", reg.intercept_, " and the weight b_1 is equal to ", reg.coef_[0])


print("the profit/loss in a city with 18 habitants is ", reg.predict([[18]]))
