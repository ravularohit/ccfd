#importing libraries
import seaborn as sns
import matplotlib.pyplot as p
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,matthews_corrcoef


#loading dataset
print("-------------------------------------\n[+]dataset loaded\n")
df = pd.read_csv("creditcard.csv")
df.dropna()

#plotting

fraud = df.loc[df['Class'] == 1]
Normal = df.loc[df['Class'] == 0]
ax = fraud.plot.scatter(x='Amount', y='Class', color='Red', label='Fraud')
Normal.plot.scatter(x='Amount', y='Class', color='Green', label='Normal', ax=ax)
p.show()
#before bar
colors = ["#0101DF", "#DF0101"]
sns.countplot('Class', data=df, palette=colors)
p.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
p.show()
#data balancing
df = df.sample(frac=1)
f_df=df.loc[df['Class']==1]
nf_df=df.loc[df['Class']==0][:492]
nd_df=pd.concat([f_df,nf_df])
df=nd_df.sample(frac=1,random_state=42)
df.head()
#after bar
colors = ["#0101DF", "#DF0101"]
sns.countplot('Class', data=df, palette=colors)
p.title('Equally Distributed Classes', fontsize=14)
p.show()
#data fabrication
print("--------------------------------------\n[+]data fabrication\n")
print("[+]dataset sample\n",df.head(),"\n-----------------------------------------------------\n")
features=df.drop('Class',axis=1)
labels=df['Class']
print("[+] labels sample\n",labels.head(),"\n--------------------------------------------------\n")
print("[+]features sample\n",features.head(),"\n-------------------------------------------------\n")


#assigning train test data
print("---------------------------------------\n[+] speration of test, train data\n")
f_train,f_test,l_train,l_test=train_test_split(features,labels,test_size=.25)


#intitializing classifiers
print("----------------------------------------\n[+]initializing classifiers\n")
#svc=SVC(probability=True,kernel='rbf',gamma=0.1,cache_size=200)
rf = RandomForestClassifier(n_estimators=10)
boost=AdaBoostClassifier(n_estimators=50,base_estimator=rf,learning_rate=1)
#boost=AdaBoostClassifier(n_estimators=50,learning_rate=1)


#fitting and predicting
boost.fit(f_train,l_train)
pred=boost.predict(f_test)
a=accuracy_score(pred,l_test)
print("-----------------------------------------\n[+]accuracy: ",a*100,"\n")
print("confusion matrix\n",confusion_matrix(l_test,pred))
print("\n",classification_report(l_test,pred))

print("\nMCC metric :",matthews_corrcoef(l_test,pred, sample_weight=None))
User= [[68207,-13.19267096,12.78597064,-9.906650021,3.320336883,-4.801175932,5.760058556,-18.75088916,-37.35344264,-0.391539744,-5.052502367,4.406805524,-4.610756477,-1.90948797,-9.072710934,-0.226074451,-6.211557482,-6.248145353,-3.149246695,0.051576119,-3.493049915,27.20283916,-8.887017141,5.303606904,-0.639434802,0.263203123,-0.10887693,1.269566355,0.939407363,1]]
out=boost.predict(User)
if(out):
	print("\nFradulent\n")
else:
	print("\ngenuine\n")

user1=[[0,-1.359807134,-0.072781173,2.536346738,1.378155224,-0.33832077,0.462387778,0.239598554,0.098697901,0.36378697,0.090794172,-0.551599533,-0.617800856,-0.991389847,-0.311169354,1.468176972,-0.470400525,0.207971242,0.02579058,0.40399296,0.251412098,-0.018306778,0.277837576,-0.11047391,0.066928075,0.128539358,-0.189114844,0.133558377,-0.021053053,149.62]]
out=boost.predict(user1)
if(out):
	print("\nFradulent\n")
else:
	print("\ngenuine\n")