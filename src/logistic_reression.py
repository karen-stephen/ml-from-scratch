#coding Logistic regression from scratch, only 1 or 2 fetures
#Gradient descent, BCE loss

import numpy as np

X = np.array([1,2,3,4,5,6])
y = np.array([0,0,0,1,1,1])

w= 0.0
b= 0.0
n= len(X)
lr = 0.01

def sigmoid(z):
  return 1/(1+np.exp(-z))

for epoch in range(100000):
  z= w*X +b
  y_pred = sigmoid(z)

  dw = 0.0
  db = 0.0

  for i in range(n):
    # error = y[i]- y_pred[i]
    error = y_pred[i] - y[i]
    dw += error*X[i]
    db += error

  dw = dw/n
  db = db/n

  w-= lr*dw
  b-= lr*db

  if epoch%100 == 0:
    loss = 0
    for i in range(n):
        loss += -1*(y[i]*np.log(y_pred[i])+ (1-y[i])* np.log(1-y_pred[i]))

    loss = loss/n

    print(f"epoch {epoch}: Loss = {loss}, w = {w}, b={b}")



s