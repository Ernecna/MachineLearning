"""
Created on Mon Jul  09/03/2024

@author:Ernecna
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('Dosyalar/odev_tenis.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
    

#encoder: Kategorik -> Numeric



from sklearn import preprocessing

le = preprocessing.LabelEncoder()

veriler2=veriler.apply(le.fit_transform)

c=veriler2.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)

havadurumu=pd.DataFrame(data=c,index=range(len(c)),columns=['o','r','s'])

son=pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1)
son=pd.concat([veriler2.iloc[:,-2:],son],axis=1)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
    

x_train, x_test,y_train,y_test = train_test_split(son.iloc[:,:-1],son.iloc[:,-1:],test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

##### reg
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


               
import statsmodels.api as sm
# B0 değerlerini eklemek yani sabitleri her bir satır için
X=np.append(arr=np.ones((14,1)).astype(int),values=son.iloc[:,:-1],axis=1)

X_liste=son.iloc[:,[0,1,2,3,4,5]].values
X_liste=np.array(X_liste,dtype=float)
model=sm.OLS(son.iloc[:,-1:],X_liste).fit()
print(model.summary())

X_liste=son.iloc[:,[1,2,3,4,5]].values
X_liste=np.array(X_liste,dtype=float)
model=sm.OLS(son.iloc[:,-1:],X_liste).fit()
print(model.summary())


## en yüksek p valuelu kaldırktan sonra
x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
    







