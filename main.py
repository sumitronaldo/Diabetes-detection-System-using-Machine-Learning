import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
#reading csv file
dbdt=pd.read_csv('diabetes.csv')
#diving data into two parts
X=dbdt.drop(columns='Outcome', axis=1)
Y=dbdt['Outcome']
#data standardisation
scalar=StandardScaler()
scalar.fit(X)
stddt=scalar.transform(X)
X=stddt
Y=dbdt['Outcome']
Xtr, Xts, Ytr, Yts = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
classifier=svm.SVC(kernel='linear')
classifier.fit(Xtr, Ytr)
Xtr_predict=classifier.predict(Xtr)
trdtac=accuracy_score(Xtr_predict, Ytr)
Xts_predict=classifier.predict(Xts)
tsdtac=accuracy_score(Xts_predict, Yts)
inputdt=(0,137,40,35,168,43.1,2.288,33)
inpnp=np.asarray(inputdt)
npinpresh=inpnp.reshape(1,-1)
stddt=scalar.transform(npinpresh)
pred=classifier.predict(stddt)
if pred[0]=='0':
	print("Non Diabetic!!!")
else:
	print("Diabetic!!")