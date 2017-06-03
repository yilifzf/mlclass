# Write code to do 10-fold cross validation and print prediction result to ten files.
# LR algorithms should be implemented, but I missed it out. check for the newton's method in zhou's book.
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

data = genfromtxt('data.csv', delimiter=',')
target = genfromtxt('targets.csv', delimiter=',')
kf = KFold(n_splits=10)
num = 0
for train, test in kf.split(data):
    # print(test)
    clf = LogisticRegression()
    clf.fit(data[train],target[train])
    pred = np.array(clf.predict(data[test]))
    rslt = np.stack((test+1,pred), axis=-1)
    num = num + 1
    filename = 'fold'+str(num)+'.csv'
    np.savetxt(filename,rslt,fmt='%d',delimiter=',')
