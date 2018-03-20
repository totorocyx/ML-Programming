#coding=utf-8
from __future__ import print_function
import pandas as pd

data_train=pd.read_csv('./breast-cancer-train.csv')
data_test=pd.read_csv('./breast-cancer-test.csv')
data_train.isnull().any()#是否有缺失值

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

#标准化数据
ss = StandardScaler()
x_train=ss.fit_transform(data_train.loc[:,['Clump Thickness','Cell Size']])
x_test=ss.transform(data_test.loc[:,['Clump Thickness','Cell Size']])
y_train=data_train.loc[:,['Type']]
y_test=data_test.loc[:,['Type']]

#newton-cg
lr=LogisticRegression(solver='newton-cg')
lr.fit(x_train,y_train)
lr_y_pred=lr.predict(x_test)
print('Acc of newton-cg:',lr.score(x_test, y_test))

#SGD
sgdc=SGDClassifier()
sgdc.fit(x_train,y_train)
sgdc_y_pred=sgdc.predict(x_test)
print('Acc of SGD:',lr.score(x_test, y_test))

#Perceptron
from sklearn.linear_model import Perceptron
ppn=Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(x_train, y_train)
ppn_y_pred=ppn.predict(x_test)

#混淆矩阵
from sklearn.metrics import classification_report
print ('Confusion Matrix of LR:\n',classification_report(y_test, \
       lr_y_pred,target_names=[u'良性', u'恶性']))
print ('Confusion Matrix of SGD:\n',classification_report(y_test, \
       sgdc_y_pred,target_names=[u'良性', u'恶性']))
print ('Confusion Matrix of Perceptron:\n',classification_report(y_test, \
       ppn_y_pred,target_names=[u'良性', u'恶性']))    

import matplotlib.pyplot as plt     
import numpy as np
import math
#分割数据集
test_neg=data_test.loc[data_test['Type']==0][['Clump Thickness','Cell Size']]
test_pos=data_test.loc[data_test['Type']==1][['Clump Thickness','Cell Size']]

#正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#系数
coef_lr=lr.coef_[0,:]
coef_sgdc=sgdc.coef_[0,:]
coef_ppn=ppn.coef_[0,:]

line_x=range(math.floor(x_test.min()),math.ceil(x_test.max()))
line_y_lr=(-lr.intercept_-line_x*coef_lr[0])/coef_lr[1]
line_y_sgdc=(-sgdc.intercept_-line_x*coef_sgdc[0])/coef_sgdc[1]
line_y_ppn=(-ppn.intercept_-line_x*coef_ppn[0])/coef_ppn[1]
#比例缩放为非标准化数据
#或使用ss.inverse_transform
lx=line_x*ss.scale_[0]+ss.mean_[0]
ly_lr=line_y_lr*ss.scale_[1]+ss.mean_[1]
ly_sgdc=line_y_sgdc*ss.scale_[1]+ss.mean_[1]
ly_ppn=line_y_ppn*ss.scale_[1]+ss.mean_[1]

#作图
plt.figure(figsize=(10,4))
#左图
plt.subplot(121)
plt.plot(lx,ly_lr,c='green',label='newton-cg')
plt.plot(lx,ly_sgdc,c='black',label='SGD')

plt.scatter(test_neg['Clump Thickness'],test_neg['Cell Size'],\
            marker='o',color='red',s=150,label=u'恶性')
plt.scatter(test_pos['Clump Thickness'],test_pos['Cell Size'],\
            marker='+',color='blue',s=100,label=u'良性')
plt.title(u'Logistic回归')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.legend(loc=2)

#右图
plt.subplot(122)
plt.plot(lx,ly_ppn,c='green',label='Perceptron')
plt.scatter(test_neg['Clump Thickness'],test_neg['Cell Size'],\
            marker='o',color='red',s=150,label=u'恶性')
plt.scatter(test_pos['Clump Thickness'],test_pos['Cell Size'],\
            marker='+',color='blue',s=100,label=u'良性')
plt.title(u'感知器模型')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.legend(loc=2)

plt.show()
