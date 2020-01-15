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
sns.set(style='white', context='notebook', palette='deep')

# %% Load data
train=pd.read_csv('train.csv')
test = pd.read_csv("test.csv")

# %%
train.head(5)

#%% rename uesr_id
train = train.rename({'uesr_id': 'user_id'}, axis=1) 
#%% factorize attrition
train['Attrition']=pd.factorize(train['Attrition'])[0]

# %% Join data
train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

# %% Check missing value
dataset.isnull().sum()
# no missing value

# %%  Delete Columns with Constant Value
dataset.drop(['Over18'],axis=1,inplace=True)
dataset.drop(['StandardHours'],axis=1,inplace=True)
dataset.drop(['EmployeeCount'],axis=1,inplace=True)



#Data Cleaning

# BusinessTravel
# %% BusinessTravel barplot
g = sns.barplot(x="BusinessTravel",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# %% Business Travel get dummies
dataset = pd.get_dummies(dataset, columns = ["BusinessTravel"], prefix="BT",drop_first=True)

# DailyRate
# %% distribution Plot
g = sns.distplot(train["DailyRate"])
# no need to clean

# Department
# %% Department barplot
g = sns.barplot(x="Department",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# %% Business Travel get dummies
dataset = pd.get_dummies(dataset, columns = ["Department"], drop_first=True)

# DistanceFromHome
# %% distplot
g = sns.distplot(dataset["DistanceFromHome"])
#%% scatter plot
g=sns.scatterplot(x="DistanceFromHome",y="Attrition",data=train)
# %%
g = sns.distplot(np.log(dataset["DistanceFromHome"]))
# %%
dataset["DistanceFromHome"]=np.log(dataset["DistanceFromHome"])

# Education
# %% bar plot
g = sns.barplot(x="Education",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# %% get dummies
dataset = pd.get_dummies(dataset, columns = ["Education"],prefix="Ed", drop_first=True)

# EducationField
# %%
g = sns.barplot(x="EducationField",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# %%
dataset = pd.get_dummies(dataset, columns = ["EducationField"],prefix="EF", drop_first=True)

# EmployeeNumber
# %%
dataset["EmployeeNumber"].head(20)
# %% drop it
train.drop(['EmployeeNumber'],axis=1,inplace=True)

# EnvironmentSatisfaction
# %%
g = sns.barplot(x="EnvironmentSatisfaction",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# only the most unsatisfactive employees are more likely to attribute
# %% reassignment
dataset['EnvironmentSatisfaction'] = dataset['EnvironmentSatisfaction'].map(lambda s: 1 if s == 1 else 0)

# Gender
# %% Bar plot
g = sns.barplot(x="Gender",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# %%
dataset = pd.get_dummies(dataset, columns = ["Gender"], drop_first=True)

# JobInvolment
# %%
g = sns.barplot(x="JobInvolvement",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# the lower the involvement the higher possibility to attribute
# %%
dataset = pd.get_dummies(dataset, columns = ["JobInvolvement"],prefix="JI", drop_first=True)

# Job Role
# %%
g = sns.barplot(x="JobRole",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# %%
dataset = pd.get_dummies(dataset, columns = ["JobRole"], prefix="JR",drop_first=True)

# JobSatisfaction
# %%
g = sns.barplot(x="JobSatisfaction",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")

# MaritalStatus
#%%
dataset = pd.get_dummies(dataset, columns = ["MaritalStatus"], prefix="MS",drop_first=True)

# Monthly Income
# %%
g = sns.distplot(dataset["MonthlyIncome"])
# %%
dataset["MonthlyIncome"]=np.log(dataset["MonthlyIncome"])

# OverTime
# %%
dataset = pd.get_dummies(dataset, columns = ["OverTime"], drop_first=True)

# PercentSalaryHike
# %%
g = sns.distplot(dataset["PercentSalaryHike"])
# %%
dataset["PercentSalaryHike"]=np.log(dataset["PercentSalaryHike"])





# Modeling
#%% Separate train dataset and test dataset
tr=dataset[:train_len].astype(int)
ts=dataset[train_len:]
ts.drop(labels=["Attrition"],axis = 1,inplace=True)

#%% Separate train features and label
tr["Attrition"]=tr["Attrition"].astype(int)
Y_train=tr["Attrition"]
X_train=tr.drop(labels = ["Attrition"],axis = 1)

#%% Simple modeling
def ModelAlg(k,r,X_train,Y_train):
    kfold = StratifiedKFold(n_splits=k)
    random_state = r
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
        cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=1))

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
    "RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})
    
    sns.set(style='white', context='notebook', palette='deep')
    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
    
    return cv_res

cv_res=ModelAlg(10,2,X_train,Y_train)



# Meta modeling with RandomForest, GradienBoosting, LR, LD
#%%
kfold = StratifiedKFold(n_splits=10)
#%% RandomForest
RFC = RandomForestClassifier()
## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsRFC.fit(X_train,Y_train)
RFC_best = gsRFC.best_estimator_
# Best score
gsRFC.best_score_

#%% Gradient boosting tunning
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsGBC.fit(X_train,Y_train)
GBC_best = gsGBC.best_estimator_
# Best score
gsGBC.best_score_



#%%  Voting Models
votingC = VotingClassifier(estimators=[('rfc', RFC_best),('gbc',GBC_best)], voting='soft', n_jobs=4)
votingC = votingC.fit(X_train, Y_train)

# %% Predicting
y1=votingC.predict_proba(ts)
gender=pd.read_csv('submission.csv')
gender['Attrition']=pd.DataFrame(y1)[1]
gender.to_csv('submission.csv',index=None)

# %%
