##!! Temel olarak, bir veya daha fazla bağımsız değişkenin bir veya daha fazla bağımlı değişken üzerindeki etkisini anlamak için kullanılır. Bağımlı değişken, tahmin etmeye çalıştığımız veya anlamaya çalıştığımız değişkendir, bağımsız değişkenler ise bu değişkeni etkileyebilecek faktörlerdir.

import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

#data upload
data = pd.read_csv('Dosyalar/satislar.csv')
#column column böldük
months=data[['Aylar']]
sales=data[['Satislar']]

sales2=data.iloc[:,:1].values
print(sales)


#test and divide
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(months,sales,test_size=0.33,random_state=0)

##Standarlaştırma normalliştirmede kullanabilirdik
#Scale kullanarak
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_Train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)

##Model İnşası
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
#Bunları alarak bir model inşa et
lr.fit(x_train,y_train)

prediction=lr.predict(x_test)
print(prediction)
#indexe göre sıralama yapılır ve her bi ay için doğru değer gelir.
x_train=x_train.sort_index()
y_train=y_train.sort_index()    

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

