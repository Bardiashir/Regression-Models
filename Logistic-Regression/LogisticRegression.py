import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt


# are imported properly. Name the first feature 'Score1', the second feature 'Score2', and the class 'y'
data = pandas.read_csv('LogisticRegressionData.csv', header=None, names=[
                       'Score1', 'Score2', 'y']) 

# Seperate the data features (score1 and Score2) from the class attribute
X = data[['Score1', 'Score2']] 
y = data['y']   

# Now we Plot the data using a scatter plot to visualize the data.
# Represent the instances with different markers of different colors based on the class labels.
m = ['o', 'x']
c = ['hotpink', '#88c999']
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker=m[int(
        data['y'][i])], color=c[int(data['y'][i])])  
fig.canvas.draw()

# Train a logistic regression classifier to predict the class labels y using the features X
regS = linear_model.LogisticRegression()  
regS.fit(X, y)  

# Now, we would like to visualize how well does the trained classifier perform on the training data
# Use the trained classifier on the training data to predict the class labels
y_pred = regS.predict(X)                    
# To visualize the classification error on the training instances, we will plot again the data. However, this time,
# the markers and colors selected will be determined using the predicted class labels
m = ['o', 'x']
c = ['red', 'blue']  # this time in red and blue
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i],
                marker=m[y_pred[i]], color=c[y_pred[i]])  
fig.canvas.draw()
plt.show()