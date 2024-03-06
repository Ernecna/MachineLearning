import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

#data upload
dataset = pd.read_csv('Dosyalar/eksikveriler.csv')


#missing values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")

age=dataset.iloc[:,1:4].values
#print(age)

## fit ile veriyi öğret tanıt
##transform ile değişikliği yap
imputer=imputer.fit(age[:,1:4])
age[:,1:4]=imputer.transform(age[:,1:4])

#################################


from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le = LabelEncoder()
country=dataset.iloc[:,0:1].values
##print(country)
country[:,0]=le.fit_transform(country[:,0])

#print(country)

ohe=OneHotEncoder(categories='auto')
country_encoded = ohe.fit_transform(country).toarray()

# Now create the DataFrame with the dense array
res = pd.DataFrame(data=country_encoded, index=range(len(country_encoded)), columns=['fr','tr','us'])
#print(res)

res2 = pd.DataFrame(data=age, index=range(len(country_encoded)), columns=['boy','kilo','yas'])
#print(res2)

gender=dataset.iloc[:,-1].values

res3=pd.DataFrame(data=gender,index=range(22),columns=['cinsiyet'])
##print(res3)


s=pd.concat([res,res2],axis=1)


s=pd.concat([res,res3],axis=1)
print(s)