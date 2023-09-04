# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary packages(Matplotlib,numpy,pandas,etc) in colab.
2. Upload the data file in colab.
3. Find MSE,MAE,RMSE using the input data.
4. Use scatter plots to plot regression lines between the data points in both training and testing set.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: B.VIJAY KUMAR
RegisterNumber: 212222230173
*/
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
df = pd.read_csv('student_scores.csv')
df
df.tail(4)
df.head(4)
X = df.iloc[:,:-1].values
X
Y = df.iloc[:,-1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state = 0)
print('X_train ',X_train)
print('Y_train ',Y_train)
print('X_test',X_test)
print('Y_test',Y_test)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
print(Y_pred)
plt.scatter(X_train,Y_train,color = "green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title('Trainijng set(H VS S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color = "green")
plt.plot(X_test,reg.predict(X_test),color="blue")
plt.title('Trainijng set(H VS S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse = mean_squared_error(Y_test,Y_pred)
mse
mae = mean_absolute_error(Y_test,Y_pred)
mae
rmse = np.sqrt(mse)
rmse
a = np.array([[10]])
Y_pred1 = reg.predict(a)
Y_pred1

```

## Output:

![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119657657/1bc2bb10-6d74-4ede-bb54-e67879d77a5e)

![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119657657/dbe27ec9-e1c4-4c4b-9655-c94a85e1eb79)

![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119657657/24d25c1f-c568-4e84-89a5-a9fbaa9f888c)

![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119657657/4f9310c7-5368-4df1-836d-64b5df0dc990)

![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119657657/c2c0a180-9d64-48a3-aa1b-e6332a01e3de)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
