import numpy as np
import pandas as pd

pd.set_option('precision',2)



import matplotlib.pyplot as plt
import seaborn as sbn



import warnings
warnings.filterwarnings('ignore')


#read and explore

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
#take a look at the training data
print train.describe()

print "\n"
print train.describe(include="all")
print "\n"
#top means whose frequency is more
#step 3) data analysis
print "\n\n",train.columns
print
print train.head()
print
print train.sample(5)

print "data type for each feature :-"
print train.dtypes
#check how many values are null/missing
print
print pd.isnull(train).sum()

#step 4) data visualization
#4.A Sex Feature
#draw a bar plot of survuval by sex
sbn.barplot(x="Sex",y="Survived",data=train)
plt.show()
print "--------------\n\n"
print train

print "--------------\n\n"
print train["Survived"]

print "--------------\n\n"
print train["Sex"]=='female'

print "--------------\n\n"
print train["Survived"][train["Sex"]=='female']


print "--------------\n\n"
print train["Survived"][train["Sex"]=='female'].value_counts()

print "--------------\n\n"
print "percentage of female who survived:", train["Survived"][train["Sex"]=='female'].value_counts(normalize=True)[1]*100
print "percentage of female who survived:", train["Survived"][train["Sex"]=='male'].value_counts(normalize=True)[1]*100

#4.B)Pclass Feature
sbn.barplot(x="Pclass",y="Survived",data=train)
plt.show()

print "percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"]==1].value_counts(normalize=True)[1]*100
print "percentage of Pclass= 2 who survived:", train["Survived"][train["Pclass"]==2].value_counts(normalize=True)[1]*100
print "percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"]==3].value_counts(normalize=True)[1]*100

#4.C)SibSp Feature
sbn.barplot(x="SibSp",y="Survived",data=train)
plt.show()

print "\n\n"
print "percentage of sibsp = 0 who survived:", train["Survived"][train["SibSp"]==0].value_counts(normalize=True)[1]*100
print "percentage of sibsp= 1 who survived:", train["Survived"][train["SibSp"]==1].value_counts(normalize=True)[1]*100
print "percentage of sibsp = 2 who survived:", train["Survived"][train["SibSp"]==2].value_counts(normalize=True)[1]*100
plt.show()


#4.D)Parch Feature
sbn.barplot(x="Parch",y="Survived",data=train)
plt.show()


#4.E)AGE Feature
#sort the ages into logical categories

train["Age"]=train["Age"].fillna(-0.5)
test["Age"]=test["Age"].fillna(-0.5)

bins=[-1,0,5,12,18,24,35,60,np.inf]
labels=['Unkown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
train['AgeGroup']=pd.cut(train["Age"],bins,labels=labels)
test['AgeGroup']=pd.cut(test["Age"],bins,labels=labels)
print train
#draw a bar plot of age vs survival
sbn.barplot(x="AgeGroup",y="Survived",data=train)
plt.show()

#4.F)Cabin Features

train["CabinBool"]=(train["Cabin"].notnull().astype('int'))
test["CabinBool"]=(test["Cabin"].notnull().astype('int'))
print "#############################\n\n"
print train

#calculate percentage of cabinbool  vs survived
print "Percentage of cabinBool=0 who survived",train["Survived"][train["CabinBool"]==0].value_counts(normalize=True)[1]*100
print "Percentage of cabinBool=1 who survived",train["Survived"][train["CabinBool"]==1].value_counts(normalize=True)[1]*100
#draw bar plot
sbn.barplot(x="CabinBool",y="Survived",data=train)
plt.show()


#5)Cleaning the data
#time to clean our data to account for missing values and unnecessary information

print test.describe(include="all")
#to know how many data are missing
print
print pd.isnull(test).sum()
print

train=train.drop(['Cabin'],axis=1)
test=test.drop(['Cabin'],axis=1)

#ticket Feature
train=train.drop(['Ticket'],axis=1)
test=test.drop(['Ticket'],axis=1)
print "\n\nNumber of people embarked in Southampton:"

print "\n\nSHAPE = ",train[train["Embarked"]=="S"].shape
print "\n\nSHAPE[0] = ",train[train["Embarked"]=="S"].shape[0]

print "\n\nNumber of people embarked in Queenstown:"

print "\n\nSHAPE = ",train[train["Embarked"]=="Q"].shape
print "\n\nSHAPE[0] = ",train[train["Embarked"]=="Q"].shape[0]

print "\n\nNumber of people embarked in cherbourg:"

print "\n\nSHAPE = ",train[train["Embarked"]=="C"].shape
print "\n\nSHAPE[0] = ",train[train["Embarked"]=="C"].shape[0]


#train=train.fillna({"Embarked":"S"})
train["Embarked"]=train["Embarked"].fillna("S")

print
print pd.isnull(test).sum()
print

combine=[train,test]
print combine[0]

for dataset in combine:
    dataset['Title']=dataset['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
    #+denotes wild card character from A-Z in capital or small


print "\n\n################################"
print train
print

print pd.crosstab(train['Title'],train['Sex'])
#replace various title with more common names
print "#################################"
for dataset in combine:
    dataset['Title']=dataset['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Sir'],'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print pd.crosstab(train['Title'],train['Sex'])
print
print train[['Title','Survived']].groupby(['Title'],as_index=False).count()
print


print "\nmap each of the title group to be numeric value"
Title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Royal":5,"Rare":6,}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(Title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
print "\nAfter replacing title with numerical value\n"
print train
print

mr_age=train[train["Title"]==1]["AgeGroup"].mode()
print "mode() of mr_age:",mr_age

miss_age=train[train["Title"]==2]["AgeGroup"].mode()
print "mode() of miss_age:",miss_age

mrs_age=train[train["Title"]==3]["AgeGroup"].mode()
print "mode() of mrs_age:",mrs_age

master_age=train[train["Title"]==4]["AgeGroup"].mode()
print "mode() of master_age:",master_age

royal_age=train[train["Title"]==5]["AgeGroup"].mode()
print "mode() of royal_age:",royal_age

rare_age=train[train["Title"]==1]["AgeGroup"].mode()
print "mode() of rare_age:",rare_age

print "\n********** train[AgeGroup][0]: \n"
for x in range(10):
    print train["AgeGroup"][x]

age_title_mapping={1:"Young Adult",2:"Student",3:"Adult",4:"Baby",5:"Adult",6:"Adult"}

for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unkown":
        train["AgeGroup"][x] = age_title_mapping[ train["Title"][x] ]

for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unkown":
        test["AgeGroup"][x] = age_title_mapping[ train["Title"][x] ]

print train
age_mapping={'Baby':1,'Child':2,'Teenager':3,'Student':4,'Young Adult':5,'Adult':6,'Senior':7}
train["AgeGroup"] = train['AgeGroup'].map(age_mapping)
test["AgeGroup"] = test['AgeGroup'].map(age_mapping)

train=train.drop(['Age'],axis=1)
test=test.drop(['Age'],axis=1)
print "\n\nAge column dropped"
print train


train=train.drop(['Name'],axis=1)
test=test.drop(['Name'],axis=1)

sex_mapping ={"male":0,"female":1}
train['Sex']=train['Sex'].map(sex_mapping)
test['Sex']=test['Sex'].map(sex_mapping)

print train


#Embarked Feature
embarked_mapping={"S":1,"C":2,"Q":3}
train['Embarked']=train['Embarked'].map(embarked_mapping)
test['Embarked']=test['Embarked'].map(embarked_mapping)
print
print train.head()
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass=test["Pclass"][x]
        test["Fare"][x]=round(train[train["Pclass"]==pclass]["Fare"].mean(),2)

#cut fare values into groups of numeric values
train['FareBand']=pd.qcut(train['Fare'],4,labels=[1,2,3,4])
test['FareBand']=pd.qcut(test['Fare'],4,labels=[1,2,3,4])


#drop fare values
train=train.drop(['Fare'],axis=1)
test=test.drop(['Fare'],axis=1)
print "\n\nFare column dropped "
print train

print
print test.head()


print "*****************************"
from sklearn.model_selection import train_test_split
input_predictors = train.drop(['Survived','PassengerId'],axis=1)
output_target=train["Survived"]
x_train,x_val,y_train,y_val=train_test_split(input_predictors,output_target,test_size=0.20,random_state=7)

from sklearn.metrics import accuracy_score
#1)Logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_val)
acc_logreg=round(accuracy_score(y_pred,y_val)*100,2)
print "MODEL-1:Accuracy of Logistic regression:",acc_logreg

#2)Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(x_train,y_train)
y_pred=gaussian.predict(x_val)
acc_gaussian=round(accuracy_score(y_pred,y_val)*100,2)
print "MODEL-2:Accuracy of gaussian nb:",acc_gaussian

#3)Support Vector Machines
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_val)
acc_svc=round(accuracy_score(y_pred,y_val)*100,2)
print "MODEL-3:Accuracy of Support Vector Machine:",acc_svc

#4)Linear SVC
from sklearn.svm import LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(x_train,y_train)
y_pred=linear_svc.predict(x_val)
acc_linear_svc=round(accuracy_score(y_pred,y_val)*100,2)
print "MODEL-4:Accuracy of linearsvc:",acc_linear_svc

#5)Perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(x_train,y_train)
y_pred=perceptron.predict(x_val)
acc_perceptron=round(accuracy_score(y_pred,y_val)*100,2)
print "MODEL-5:Accuracy of decision tree:",acc_perceptron

#6)Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train,y_train)
y_pred=decisiontree.predict(x_val)
acc_decisiontree=round(accuracy_score(y_pred,y_val)*100,2)
print "MODEL-6:Accuracy of Decision Tree:",acc_decisiontree

#7)Random Forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()
randomforest.fit(x_train,y_train)
y_pred=randomforest.predict(x_val)
acc_randomforest=round(accuracy_score(y_pred,y_val)*100,2)
print "MODEL-7:Accuracy of Random Forrest:",acc_randomforest

#8)KNN or K-Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred=knn.predict(x_val)
acc_knn=round(accuracy_score(y_pred,y_val)*100,2)
print "MODEL-8:Accuracy of knn:",acc_knn


#9)Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(x_train,y_train)
y_pred=sgd.predict(x_val)
acc_sgd=round(accuracy_score(y_pred,y_val)*100,2)
print "MODEL-9:Accuracy of SGD:",acc_sgd


#10)Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train,y_train)
y_pred=gbc.predict(x_val)
acc_gbc=round(accuracy_score(y_pred,y_val)*100,2)
print "MODEL-10:Accuracy of GBC:",acc_gbc


#lets compare the accuracies of each model
models=pd.DataFrame({
    'Model':['Logistic Regression','Gaussian Naive Bayes','Support Vector Machines','Linear SVC'
             ,'Perceptron','Decision Tree','Random Forest','KNN','Stochastic Gradient Descent'
                ,'Gradient Boosting Classifier'],
    'Score':[acc_logreg,acc_gaussian,acc_svc,acc_linear_svc,acc_perceptron,acc_decisiontree
             ,acc_randomforest,acc_knn,acc_sgd,acc_gbc]
})
print
print models.sort_values(by='Score',ascending=False)

#Creating submission file

ids=test['PassengerId']
predictions=gbc.predict(test.drop('PassengerId',axis=1))
output=pd.DataFrame({'PassengerId':ids,'Survived':predictions})
output.to_csv('Submission.csv',index=False)
print "All the predictions are done"
print "All predictions exported to submission.csv file"
print output