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
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


train=pd.read_csv('train.csv')

# %%
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

# %%

#train[train['Sex']=='male']=1
#train[train['Sex']=='female']=0
Sex=pd.get_dummies(train['Sex'],drop_first=True)
Embarked=pd.get_dummies(train['Embarked'],drop_first=True)
Fare=train['Fare']
NFare=(Fare-Fare.min())/(Fare.max()-Fare.min())
Pclass=pd.get_dummies(train['Pclass'],drop_first=True)
SibSp=train['SibSp']
Parch=train['Parch']

Male_Avg1 = np.mean(test[test['Sex'] == 0].Age)               # 0 is male 1 is female by factorize function
Female_Avg1= np.mean(test[test['Sex'] == 1].Age) 

# Now lets fill them 
test['Age'] = test[['Sex','Age']].apply(lambda x : Fillme(x,Male_Avg1,Female_Avg1) , axis = 1) 
Age=train['Age']
Age.fillna(int(Age.mode()),inplace=True)
NAge=(Age-Age.min())/(Age.max()-Age.min())

X=pd.concat([Sex,Embarked,NFare,Pclass,SibSp,Parch,NAge],axis=1)
y=train['Survived']


# %%
test=pd.read_csv('test.csv')

test['Embarked'].fillna(method='ffill',inplace=True)
test.drop(['Name'],axis = 1,inplace = True)
test.drop(['Ticket'],axis=1,inplace=True)
test.drop(['Cabin'],axis=1,inplace=True)
test['Sex']=pd.factorize(test['Sex'])[0]
test['Embarked']=pd.factorize(test['Embarked'])[0]


Sex1=pd.get_dummies(test['Sex'],drop_first=True)
Embarked1=pd.get_dummies(test['Embarked'],drop_first=True)
Fare1=test['Fare']
Fare1.fillna(int(Fare1.mode()),inplace=True)
NFare1=(Fare1-Fare1.min())/(Fare1.max()-Fare1.min())
Pclass1=pd.get_dummies(test['Pclass'],drop_first=True)
SibSp1=test['SibSp']
Parch1=test['Parch']
Age1=test['Age']
Age1.fillna(int(Age.mode()),inplace=True)
NAge1=(Age1-Age1.min())/(Age1.max()-Age1.min())


X1=pd.concat([Sex1,Embarked1,NFare1,Pclass1,SibSp1,Parch1,NAge1],axis=1)




#%%
clf=LogisticRegression(random_state=0).fit(X,y)
clf.score(X,y)
#%%
clf_tree=DecisionTreeClassifier(random_state=0).fit(X,y)
clf_tree.score(X,y)
#%%
clf_svm=LinearSVC(random_state=0, tol=1e-5).fit(X,y)
clf_svm.score(X,y)
#%%
clf_random=RandomForestClassifier(max_depth=10, random_state=0).fit(X,y)
clf_random.score(X,y)
#%%
clf_ada = AdaBoostClassifier(random_state=0,n_estimators=100).fit(X,y)
clf_ada.score(X,y)




#%%
def ModelAlg(k,r,X_train,Y_train):
    kfold = StratifiedKFold(n_splits=k)
    random_state = r,
    classifiers = []
    classifiers.append(SVC(random_state=random_state))
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(MLPClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state = random_state))
    classifiers.append(LinearDiscriminantAnalysis())

    cv_results = []
    for classifier in classifiers :
        cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
    "RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})
    
    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
    
    return cv_res




# %%
y1=clf_tree.predict(X1)
gender=pd.read_csv('gender_submission.csv')
gender['Survived']=y1
gender.to_csv('submission.csv',index=None)

# %%
