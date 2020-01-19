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
# %% Department get dummies
dataset = pd.get_dummies(dataset, columns = ["Department"], drop_first=True)

# DistanceFromHome
# %% distplot
g = sns.distplot(dataset["DistanceFromHome"])
#%% scatter plot
g=sns.scatterplot(x="DistanceFromHome",y="Attrition",data=train)
# %%
g = sns.distplot(np.log(dataset["DistanceFromHome"]))
# %% Log DistanceFromHome 
dataset["DistanceFromHome"]=np.log(dataset["DistanceFromHome"])



# Education
# %% bar plot
g = sns.barplot(x="Education",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# %% Education get dummies
dataset = pd.get_dummies(dataset, columns = ["Education"],prefix="Ed", drop_first=True)

# EducationField
# %%
g = sns.barplot(x="EducationField",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# %% EducationField get dummies
dataset = pd.get_dummies(dataset, columns = ["EducationField"],prefix="EF", drop_first=True)

# EmployeeNumber
# %%
dataset["EmployeeNumber"].head(20)
# %% drop EmployeeNumber
dataset.drop(['EmployeeNumber'],axis=1,inplace=True)

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
# %% Gender get dummies
dataset = pd.get_dummies(dataset, columns = ["Gender"], drop_first=True)

# JobInvolment
# %%
g = sns.barplot(x="JobInvolvement",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# the lower the involvement the higher possibility to attribute
# %% JobInvolment get dummies
dataset = pd.get_dummies(dataset, columns = ["JobInvolvement"],prefix="JI", drop_first=True)

# JobRole
# %%
g = sns.barplot(x="JobRole",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# %% Job Role get dummies
dataset = pd.get_dummies(dataset, columns = ["JobRole"], prefix="JR",drop_first=True)

# JobSatisfaction
# %% JobSatisfaction
g = sns.barplot(x="JobSatisfaction",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")

# MaritalStatus
#%% MaritalStatus
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
# %% PercentSalaryHike
dataset["PercentSalaryHike"]=np.log(dataset["PercentSalaryHike"])

# PerformanceRating
#%% Plot
g = sns.distplot(dataset["PerformanceRating"])
#%% Performancerating Statistic
dataset["PerformanceRating"].describe()
# Only two values
#%% PerformanceRating get dummies
dataset = pd.get_dummies(dataset, columns = ["PerformanceRating"], prefix="PR",drop_first=True)

# RelationshipSatisfaction
#%% RelationshipSatisfaction barplot
g = sns.barplot(x="RelationshipSatisfaction",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# %% Job Role get dummies
dataset = pd.get_dummies(dataset, columns = ["RelationshipSatisfaction"], prefix="RS",drop_first=True)

# StockOptionLevel
#%% StockOptionLevel distPlot
g = sns.distplot(dataset["StockOptionLevel"])
#%% barplot
g = sns.barplot(x="StockOptionLevel",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# %% StockOptionLevel get dummies
dataset = pd.get_dummies(dataset, columns = ["StockOptionLevel"], prefix="SOL",drop_first=True)

# TotalWorkingYears
#%% TotalWorkingYears distPlot
g = sns.distplot(dataset["TotalWorkingYears"])
#%% log TotalWorkingYears
# dataset["LogTotalWorkingYears"]=np.log(dataset["TotalWorkingYears"])

# TrainingTimesLastYear
#%% TrainingTimesLastYear barplot
g = sns.barplot(x="TrainingTimesLastYear",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
# %% StockOptionLevel get dummies
dataset = pd.get_dummies(dataset, columns = ["TrainingTimesLastYear"], prefix="TT",drop_first=True)

# WorkLifeBalance
#%% WorkLifeBalance barplot
g = sns.barplot(x="WorkLifeBalance",y="Attrition",data=train)
g = g.set_ylabel("Attrition Probability")
#%% Reassign
dataset['WorkLifeBalance'] = dataset['WorkLifeBalance'].map(lambda s: 1 if s == 1 else 0)

# YearsAtCompany
#%% YearsAtCompany distPlot
g = sns.distplot(dataset["YearsAtCompany"])
#%% log YearsAtCompany
# dataset["YearsAtCompany"][dataset["YearsAtCompany"]>dataset["YearsAtCompany"].quantile(0.95)]=dataset["YearsAtCompany"].quantile(0.95)
dataset["YearsAtCompany"]=np.log1p(dataset["YearsAtCompany"])

#%% YearsInCurrentRole distPlot
g = sns.distplot(dataset["YearsInCurrentRole"])
#%% log YearsInCurrentRole 
#dataset["YearsInCurrentRole"][dataset["YearsInCurrentRole"]>dataset["YearsInCurrentRole"].quantile(0.95)]=dataset["YearsInCurrentRole"].quantile(0.95)
dataset["YearsInCurrentRole"]=np.log1p(dataset["YearsInCurrentRole"])

#%% YearsSinceLastPromotion distPlot
g = sns.distplot(dataset["YearsSinceLastPromotion"])
#%% YearsSinceLastPromotion
# dataset["YearsSinceLastPromotion"][dataset["YearsSinceLastPromotion"]>dataset["YearsSinceLastPromotion"].quantile(0.93)]=dataset["YearsSinceLastPromotion"].quantile(0.93)
dataset["YearsSinceLastPromotion"]=np.log1p(dataset["YearsSinceLastPromotion"])

#%% YearsWithCurrManager distPlot
g = sns.distplot(dataset["YearsWithCurrManager"])
#%% log YearsWithCurrManager
dataset["YearsWithCurrManager"]=np.log1p(dataset["YearsWithCurrManager"])

#%%
# dataset[dataset==np.isinf]=0

#%% Drop user_id
dataset.drop(['user_id'],axis=1,inplace=True)







# Modeling
#%% Separate train dataset and test dataset
tr=dataset[:train_len]
ts=dataset[train_len:]
ts.drop(labels=["Attrition"],axis = 1,inplace=True)

#%% Separate train features and label
tr["Attrition"]=tr["Attrition"].astype(int)
Y_train=tr["Attrition"]
X_train=tr.drop(labels = ["Attrition"],axis = 1)
X_train=(X_train-X_train.min())/(X_train.max()-X_train.min())
ts=(ts-ts.min())/(ts.max()-ts.min())




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

cv_res



# Meta modeling with RandomForest,ExtraTrees, GradienBoosting, LR, LDA, MP
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

#%% ExtraTrees
ETC = ExtraTreesClassifier()
## Search grid for optimal parameters
et_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsETC = GridSearchCV(ETC,param_grid = et_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsETC.fit(X_train,Y_train)
ETC_best = gsETC.best_estimator_
# Best score
gsETC.best_score_

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



#%% LinearDiscriminantAnalysis tunning
LDC = LinearDiscriminantAnalysis()
ld_param_grid = {'solver': ['eigen','svd'],
              }
gsLDC = GridSearchCV(LDC,param_grid = ld_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsLDC.fit(X_train,Y_train)
LDC_best = gsLDC.best_estimator_
# Best score
gsLDC.best_score_


#%% LogisticRegression tunning
LRC = LogisticRegression()
lr_param_grid = {'solver': ['newton-cg','lbfgs', 'liblinear', 'sag', 'saga']
              }
gsLRC = GridSearchCV(LRC,param_grid = lr_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsLRC.fit(X_train,Y_train)
LRC_best = gsLRC.best_estimator_
# Best score
gsLRC.best_score_


#%% MLPClassifier tunning
MLPC = MLPClassifier()
mlpc_param_grid = {'activation':['logistic'],
    'solver': ['sgd','adam'],

              }
gsMLPC = GridSearchCV(MLPC,param_grid = mlpc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsMLPC.fit(X_train,Y_train)
MLPC_best = gsMLPC.best_estimator_
# Best score
gsMLPC.best_score_


#%%
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


#%%
g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)
#%%
g = plot_learning_curve(gsETC.best_estimator_,"ExtraTrees mearning curves",X_train,Y_train,cv=kfold)
#%%
g = plot_learning_curve(gsGBC.best_estimator_,"GB mearning curves",X_train,Y_train,cv=kfold)
#%%
g = plot_learning_curve(gsLRC.best_estimator_,"LR mearning curves",X_train,Y_train,cv=kfold)
#%%
g = plot_learning_curve(gsLDC.best_estimator_,"LD mearning curves",X_train,Y_train,cv=kfold)
#%%
g = plot_learning_curve(gsMLPC.best_estimator_,"MLP mearning curves",X_train,Y_train,cv=kfold)






#%%  Voting Models
votingC = VotingClassifier(estimators=[('MLPC',MLPC_best),('ld',LDC_best),('lr',LRC_best)], voting='soft', n_jobs=4)
votingC = votingC.fit(X_train, Y_train)
votingC.score(X_train,Y_train)

# %% Predicting
y1=votingC.predict_proba(ts)
result = pd.DataFrame()
result['user_id'] = test['user_id']
result['Attrition'] = pd.DataFrame(y1)[1]
result.to_csv('submission.csv',index=None)

# %%