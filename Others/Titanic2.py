#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

train=pd.read_csv('train.csv')

# %%

#train[train['Sex']=='male']=1
#train[train['Sex']=='female']=0
train['Embarked'].fillna(method='ffill',inplace=True)
train.drop(['Name'],axis = 1,inplace = True)
train.drop(['Ticket'],axis=1,inplace=True)
train.drop(['Cabin'],axis=1,inplace=True)
train['Sex']=pd.factorize(train['Sex'])[0]
train['Embarked']=pd.factorize(train['Embarked'])[0]

Diag = sns.PairGrid(train)
Diag.map(plt.scatter)
plt.show()


#%%
def Fillme(Value,Male_Avg,Female_Avg):         # Helper Function 
    Sex = Value[0]        # Value[0] is the sex attribute
    Age = Value[1]        # Value[1] is the Age attribute
    
    if pd.isnull(Age):
        if Sex == 0:      # If its Male else Female
            return Male_Avg
        else:
            return Female_Avg
    else:
        return Age

#%%
Male_Avg = np.mean(train[train['Sex'] == 0].Age)               # 0 is male 1 is female by factorize function
Female_Avg = np.mean(train[train['Sex'] == 1].Age) 

# Now lets fill them 
train['Age'] = train[['Sex','Age']].apply(lambda x : Fillme(x,Male_Avg,Female_Avg) , axis = 1) 


#%%
y=train['Survived']
train.drop(['Survived'],axis=1,inplace=True)
X=(train-train.min())/(train.max()-train.min())




# %%
test=pd.read_csv('test.csv')

Sex1=pd.get_dummies(test['Sex'],drop_first=True)
Embarked1=pd.get_dummies(test['Embarked'],drop_first=True)
Fare1=test['Fare']
NFare=(Fare1-Fare1.min())/(Fare1.max()-Fare1.min())
Fare1.fillna(int(Fare.mode()),inplace=True)
Pclass1=pd.get_dummies(test['Pclass'],drop_first=True)
SibSp1=test['SibSp']
Parch1=test['Parch']
Age1=test['Age']
Age1.fillna(int(Age.mode()),inplace=True)
NAge=(Age1-Age1.min())/(Age1.max()-Age1.min())


X1=pd.concat([Sex1,Embarked1,Fare1,Pclass1,SibSp1,Parch1,Age1],axis=1)




#%%
clf=LogisticRegression(random_state=0).fit(X,y)
clf.score(X,y)

clf_tree=DecisionTreeClassifier(random_state=0).fit(X,y)
clf_tree.score(X,y)

clf_svm=LinearSVC(random_state=0, tol=1e-5).fit(X,y)
clf_svm.score(X,y)

clf_random=RandomForestClassifier(max_depth=10, random_state=0).fit(X,y)
clf_random.score(X,y)

clf_ada = AdaBoostClassifier(random_state=0,n_estimators=100).fit(X,y)
clf_ada.score(X,y)






# %%
y1=clf_ada.predict(X1)
gender=pd.read_csv('gender_submission.csv')
gender['Survived']=y1

#%%
gender.to_csv('submission.csv',index=None)

# %%
