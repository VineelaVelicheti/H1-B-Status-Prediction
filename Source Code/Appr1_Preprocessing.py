
# coding: utf-8

# In[76]:


import pandas as pd
import numpy as np
from collections import Counter
import time
import matplotlib.pyplot as plt


# In[77]:


data = pd.read_csv("/Users/vineevineela/Desktop/H-1B_data.csv")


# In[3]:


data.dtypes


# In[4]:


data.shape


# In[5]:


#Considering a subset of 100k from the full data
data = data.iloc[0:100000,:]


# In[6]:


#data.head(20)


# In[7]:


#Dealing with individual columns

#Filling AGENT_REPRESENTING_EMPLOYER
data['AGENT_REPRESENTING_EMPLOYER'] = np.where(data['AGENT_ATTORNEY_NAME'].isnull(), 'no', 'yes')
data['AGENT_REPRESENTING_EMPLOYER'].value_counts()

#dropping AGENT_ATTORNEY_NAME
data = data.drop(['AGENT_ATTORNEY_NAME'],axis=1)

#Creating dummies for AGENT_REPRESENTING_EMPLOYER
data['AGENT_REPRESENTING_EMPLOYER'] = data['AGENT_REPRESENTING_EMPLOYER'].fillna('blank')
agent_rep_dummy = pd.get_dummies(data['AGENT_REPRESENTING_EMPLOYER'],prefix = 'agent_rep',sparse = True)
data = pd.concat([data,agent_rep_dummy],axis=1)

data = data.drop(['AGENT_REPRESENTING_EMPLOYER'],axis=1)


# In[8]:


#filling EMPLOYER_COUNTRY = USA when EMPLOYER_STATE is one of the 56 states
l= ('AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','DC','VI','GU','MP','PR')
X = len(l)
for i in range(X):
    data.loc[data.EMPLOYER_STATE == l[i], 'EMPLOYER_COUNTRY'] = 'UNITED STATES OF AMERICA'
    

#filtering rows which dont have country as "USA" and drop the column EMPLOYER_COUNTRY
data = data[(data[['EMPLOYER_COUNTRY']] == 'UNITED STATES OF AMERICA').all(axis=1)]
data = data.drop(['EMPLOYER_COUNTRY'],axis=1)


# In[9]:


# EMPLOYER_STATE is a categorical. So dummies are used
estate_dummy = pd.get_dummies(data['EMPLOYER_STATE'],prefix = 'emp_state',sparse = True)
data = pd.concat([data,estate_dummy],axis=1)

#dropping EMPLOYER_STATE
data = data.drop(['EMPLOYER_STATE'],axis=1)


# In[10]:


#DURATION is derived from given two columns: EMPLOYMENT_END_DATE,EMPLOYMENT_START_DATE
#convert to timedate type
data['EMPLOYMENT_END_DATE'] = pd.to_datetime(data['EMPLOYMENT_END_DATE'],errors='coerce')
data['EMPLOYMENT_START_DATE'] = pd.to_datetime(data['EMPLOYMENT_START_DATE'],errors='coerce')


# In[11]:


data['DURATION'] = (data['EMPLOYMENT_END_DATE']-data['EMPLOYMENT_START_DATE'])
data['DURATION'] = (data['DURATION']/np.timedelta64(1, 'Y'))


# In[12]:


#dropping EMPLOYMENT_END_DATE,EMPLOYMENT_START_DATE
data = data.drop(['EMPLOYMENT_END_DATE','EMPLOYMENT_START_DATE'],axis=1)


# In[13]:


#H1-B Dependent is a categorical with some nan values. 
#Filling nan with a new category "Blank" and creating dummies
data['H1B_DEPENDENT'] = data['H1B_DEPENDENT'].fillna('blank')
h1bdep_dummy = pd.get_dummies(data['H1B_DEPENDENT'],prefix = 'H1b dependent',sparse = True)
data = pd.concat([data,h1bdep_dummy],axis=1)

data = data.drop(['H1B_DEPENDENT'],axis=1)


# In[14]:


#PREVAILING_WAGE, PW_SOURCE, PW_UNIT_OF_PAY, WAGE_RATE_OF_PAY_FROM, WAGE_RATE_OF_PAY_TO, WAGE_UNIT_OF_PAY
#convert prevailing wage to same scale : "year"
data.loc[data.PW_UNIT_OF_PAY == 'Hour', 'PREVAILING_WAGE'] = data['PREVAILING_WAGE']*2080
data.loc[data.PW_UNIT_OF_PAY == 'Month', 'PREVAILING_WAGE'] = data['PREVAILING_WAGE']*12
data.loc[data.PW_UNIT_OF_PAY == 'Week', 'PREVAILING_WAGE'] = data['PREVAILING_WAGE']*52
data.loc[data.PW_UNIT_OF_PAY == 'Bi-Weekly', 'PREVAILING_WAGE'] = data['PREVAILING_WAGE']*24

data.loc[data.WAGE_UNIT_OF_PAY == 'Hour', 'WAGE_RATE_OF_PAY_FROM'] = data['WAGE_RATE_OF_PAY_FROM']*2080
data.loc[data.WAGE_UNIT_OF_PAY == 'Month', 'WAGE_RATE_OF_PAY_FROM'] = data['WAGE_RATE_OF_PAY_FROM']*12
data.loc[data.WAGE_UNIT_OF_PAY == 'Week', 'WAGE_RATE_OF_PAY_FROM'] = data['WAGE_RATE_OF_PAY_FROM']*52
data.loc[data.WAGE_UNIT_OF_PAY == 'Bi-Weekly', 'WAGE_RATE_OF_PAY_FROM'] = data['WAGE_RATE_OF_PAY_FROM']*24


# In[15]:


#creating new columns "INCREMENT" , "INCREMENT_FROM_PREV_WAGE"
data['INCREMENT_FROM_PREV_WAGE'] = data['WAGE_RATE_OF_PAY_FROM'] - data['PREVAILING_WAGE']
data['INCREMENT'] = np.where(data['WAGE_RATE_OF_PAY_FROM'] >= data['PREVAILING_WAGE'],'True','False')

data = data.drop(['WAGE_RATE_OF_PAY_FROM','WAGE_RATE_OF_PAY_TO','PW_UNIT_OF_PAY','WAGE_UNIT_OF_PAY',
                  'PREVAILING_WAGE'],1)


# In[16]:


#PW_SOURCE
pwsource_dummy = pd.get_dummies(data['PW_SOURCE'],prefix = 'pw_source',sparse = True)
data = pd.concat([data,pwsource_dummy],axis=1)

data = data.drop(['PW_SOURCE'],1)


# In[17]:


data = data.dropna()


# In[18]:


#SOC_CODE
#since SOC_CODE has many categories and sub categories only categories are dealt with using one hot encoding
new_soc_code = data.SOC_CODE.apply(lambda a: int(str(a)[:2]))
soc_dummy = pd.get_dummies(new_soc_code, prefix = 'soc_code',sparse = True)
data = pd.concat([data,soc_dummy],axis=1)

data = data.drop(['SOC_CODE'],1)


# In[19]:


#VISA_CLASS
visa_dummy = pd.get_dummies(data['VISA_CLASS'],prefix = 'visa',sparse = True)
data = pd.concat([data,visa_dummy],axis=1)

data = data.drop(['VISA_CLASS'],1)


# In[20]:


#WILLFUL_VIOLATOR
data['WILLFUL_VIOLATOR'] = data['WILLFUL_VIOLATOR'].fillna('blank')
wf_dummy = pd.get_dummies(data['WILLFUL_VIOLATOR'],prefix = 'willful violator')
data = pd.concat([data,wf_dummy],axis=1)

data = data.drop(['WILLFUL_VIOLATOR'],1)


# In[21]:


#WORKSITE_STATE 
wstate_dummy = pd.get_dummies(data['WORKSITE_STATE'],prefix = 'work_state',sparse = True)
data = pd.concat([data,wstate_dummy],axis=1)

data = data.drop(['WORKSITE_STATE'],1)


# In[23]:


#EMPLOYER_NAME has too many valuesto do one hot encoding. To retain information 2 numeric columns are created
#from the given column.
#ACCEPTED_RATE and APP_COUNT

#app_count gives total no. of applications applied by each employer
app_count = Counter(data['EMPLOYER_NAME'])

#accepted_count gives total no. of accepted applications by each employer
filter_df = data[(data[['CASE_STATUS']] == 'CERTIFIED').all(axis=1)]
accepted_count = Counter(filter_df['EMPLOYER_NAME'])


accepted_rate = []
ename = []
for key1, value1 in app_count.iteritems():
    for key2, value2 in accepted_count.iteritems():
        if(key1 == key2):
            acc_rate =  round((float(value2)/value1)*100)
            accepted_rate.append(acc_rate)
            ename.append(key1)
            break            

app_details = dict(zip(ename,accepted_rate))
for key in app_details.iterkeys():
    data.loc[data.EMPLOYER_NAME == key, 'ACCEPTED_RATE'] = app_details[key]

for key in app_count.iterkeys():
    data.loc[data.EMPLOYER_NAME == key, 'APP_COUNT'] = app_count[key]

data['ACCEPTED_RATE'] = data['ACCEPTED_RATE'].fillna(value=0)
del(filter_df)


# In[24]:


#delete column EMPLOYER_NAME
data = data.drop(['EMPLOYER_NAME'],axis=1)


# In[25]:


data = data.dropna()


# In[26]:


from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)

data['INCREMENT'] = lb.fit_transform(data["INCREMENT"])
data['CASE_STATUS'] = lb.fit_transform(data["CASE_STATUS"])


# In[27]:


#data.dtypes


# In[28]:


#splitting the data frame to x and y
target = pd.DataFrame(data['CASE_STATUS'])
data = data.drop(['CASE_STATUS'],1)


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2)


# In[30]:


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(X_train)
train_data = normalizer.transform(X_train)
test_data = normalizer.transform(X_test)


# In[31]:


#Dimensionality reduction : PCA

from sklearn.decomposition import PCA
import time

start_time = time.clock()

pca = PCA(n_components=100)
pca = pca.fit(train_data)

Train_PCA = pca.transform(train_data)
print(Train_PCA.shape)

Test_PCA = pca.transform(test_data)
print(Test_PCA.shape)

print("Time Taken in seconds:", time.clock() - start_time) 


# In[32]:


#GradientBoost

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

start_time = time.clock()

c2 = GradientBoostingClassifier(n_estimators=500)
c2.fit(Train_PCA,Y_train)
GradB_Y =c2.predict(Test_PCA)

from sklearn.metrics import accuracy_score
Grad_Accuracy = accuracy_score(Y_test, GradB_Y)
print("Gradient Boost accuracy = ",Grad_Accuracy)

from sklearn.metrics import f1_score
Grad_F1 = f1_score(Y_test,GradB_Y, average= 'weighted')
print("Gradient Boost F1Score = ",Grad_F1)

print("Time Taken in seconds:", time.clock() - start_time) 


# In[33]:


#MLP CLassifier

from sklearn.neural_network import MLPClassifier

start_time = time.clock()

mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(15,), random_state=1)
Mlp_model = mlp.fit(Train_PCA,Y_train)
pred_mlp = Mlp_model.predict(Test_PCA)

from sklearn.metrics import accuracy_score
MLP_Accuracy = accuracy_score(Y_test, pred_mlp)
print("MLP Accuracy=",MLP_Accuracy)

from sklearn.metrics import f1_score
MLP_F1 = f1_score(Y_test, pred_mlp, average= 'weighted') 
print("MLP F1Score=",MLP_F1)

print("Time Taken in seconds:", time.clock() - start_time) 


# In[34]:


#SGD Classifier

from sklearn.linear_model import SGDClassifier

start_time = time.clock()

SGD_model = SGDClassifier().fit(Train_PCA,Y_train)
pred_sgd = SGD_model.predict(Test_PCA)

from sklearn.metrics import accuracy_score
SGD_Accuracy = accuracy_score(Y_test, pred_sgd)
print("SGD Accuracy=",SGD_Accuracy)

from sklearn.metrics import f1_score
SGD_F1 = f1_score(Y_test, pred_sgd, average= 'weighted') 
print("SGD F1Score=",SGD_F1)

print("Time Taken in seconds:", time.clock() - start_time) 


# In[41]:


#plotting a graph comparing various Accuracies

x_axis = np.arange(3)
y_axis = [Grad_Accuracy, MLP_Accuracy,  SGD_Accuracy]

plt.bar(x_axis, y_axis, width=0.5)
plt.xticks(x_axis+0.5/10.,('GrdBst','MLP','SGD'))
plt.ylabel('Accuracy')

plt.show()


# In[35]:


#plotting a graph comparing various Accuracies

x_axis = np.arange(3)
y_axis = [Grad_Accuracy, MLP_Accuracy,  SGD_Accuracy]
plt.plot(x_axis,y_axis)
plt.xticks(x_axis+0.5/10.,('GrdBst','MLP','SGD'))
plt.ylabel('Accuracy')
plt.yticks(np.arange(0,1,15))
plt.show()


# In[42]:


#plotting a graph comparing various Accuracies

x_axis = np.arange(3)
y_axis = [Grad_F1, MLP_F1,  SGD_F1]

plt.bar(x_axis, y_axis, width=0.5)
plt.xticks(x_axis+0.5/10.,('GrdBst','MLP','SGD'))
plt.ylabel('F1 Score')

plt.show()


# In[38]:


#plotting a graph comparing various Accuracies

x_axis = np.arange(3)
y_axis = [Grad_F1, MLP_F1,  SGD_F1]
plt.plot(x_axis,y_axis)
plt.xticks(x_axis+0.5/10.,('GrdBst','MLP','SGD'))
plt.ylabel('Accuracy')
plt.yticks(np.arange(0,1))
plt.show()


# In[51]:


x_axis = np.arange(9)
y_axis = [Dec_F1 , Grad_F1, Bag_F1, 
          KNN_F1, RF_F1, AdaB_F1, EXT_F1, MLP_F1,  SGD_F1]
plt.plot(x_axis,y_axis)
plt.xticks(x_axis+0.5/10.,('DecTree','GrdBst',
           'Bag','KNN','RF',
            'AdaBst','ExtTree','MLP','SGD'))
plt.ylabel('Accuracy')
plt.yticks(np.arange(0,1,15))
plt.show()


# In[49]:


#Cross_validation for top 3 algorithms with 5 folds
from sklearn.model_selection import cross_val_score

A1_score = cross_val_score(c2,X_train,Y_train,cv=5).mean()
A2_score = cross_val_score(mlp, X_train, Y_train, cv=5).mean()
A3_score = cross_val_score(SGD_model, X_train, Y_train, cv=5).mean()

print('Gradient Boost cross validation score', A1_score)
print('MNL cross validation score', A2_score)
print('SGD cross validation score', A3_score)


# In[51]:


#Plotting bar chart for cv score with 5 folds

x_axis = np.arange(3)
y_axis = [A1_score, A2_score, A3_score]

plt.bar(x_axis, y_axis, width=0.5)
plt.xticks(x_axis+0.5/10.,('GB','MLP','SGD'))
plt.ylabel('Cross-Val Accuracy')

plt.show()


# In[52]:


#Cross_validation for top 3 algorithms with 10 folds
from sklearn.model_selection import cross_val_score

a1_score = cross_val_score(c2,X_train,Y_train,cv=10).mean()
a2_score = cross_val_score(mlp, X_train, Y_train, cv=10).mean()
a3_score = cross_val_score(SGD_model, X_train, Y_train, cv=10).mean()

print('Gradient Boost cross validation score', a1_score)
print('MNL cross validation score', a2_score)
print('SGD cross validation score', a3_score)


# In[53]:


#Plotting bar chart for cv score with 10 folds

x_axis = np.arange(3)
y_axis = [a1_score, a2_score, a3_score]

plt.bar(x_axis, y_axis, width=0.5)
plt.xticks(x_axis+0.5/10.,('GB','MLP','SGD'))
plt.ylabel('Cross-Val Accuracy')

plt.show()


# In[ ]:


#Fine tuning top 3 classifiers using GridSearchCV


# In[58]:


#A1 - 
#considering various parameters to check for better accuracy using grid search cv

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import time

start = time.time()

parameters = {'n_estimators':range(20,81,10)}

c2g = GradientBoostingClassifier()

grid_a1 = GridSearchCV(c2g,parameters,cv=10,scoring='f1_weighted')
grid_a1.fit(X_train,Y_train)


print('Gradient Boosting')
print(grid_a1.best_score_)

print('Time_Elapsed: ', (time.time()-start))


# In[60]:


#A2 - 
#considering various parameters to check for better accuracy using grid search cv

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

import time

start = time.time()

parameters ={"solver":['sgd'],
            "hidden_layer_sizes":(15,),
            "random_state":[1]}
            
mlpg = MLPClassifier()

grid_a2 = GridSearchCV(mlpg,parameters,cv=10,scoring='accuracy')
grid_a2.fit(X_train,Y_train)

print('MLP')
print(grid_a2.best_score_)

print('Time_Elapsed: ', (time.time()-start))


# In[62]:


#A3 - 
#considering various parameters to check for better accuracy using grid search cv

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
import time

start = time.time()

parameters ={"loss":['log'],
            
            "n_jobs":[-1]}
            

SGD_model = SGDClassifier()

grid_a3 = GridSearchCV(SGD_model,parameters,cv=10,scoring='accuracy')
grid_a3.fit(X_train,Y_train)

print('SGD')
print(grid_a3.best_score_)
print('Time_Elapsed: ', (time.time()-start))


# In[63]:


#Plot graph

x_axis = np.arange(3)
y_axis = [grid_a1.best_score_, grid_a2.best_score_, grid_a3.best_score_]

plt.bar(x_axis, y_axis, width=0.5)
plt.xticks(x_axis+0.5/10.,('GB','MLP','SGD'))
plt.ylabel('FineTunedAcuuracy')
plt.show()


# In[65]:


#Displaying all metrics  to compare classification algorithms

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print('Gradient Boosting Classifier Result:')
pred_a1 = grid_a1.predict(X_test)

acc1 = metrics.accuracy_score(Y_test,pred_a1)
print('Prediction Accuracy')
print(acc1)

facc1 = metrics.f1_score(Y_test,pred_a1,average= 'weighted')
print('Prediction F1 Score')
print(facc1)

print('Report')
print(classification_report(Y_test,pred_a1))

print('Confusion Matrix')
print(confusion_matrix(Y_test, pred_a1))



# In[66]:


#Displaying all metrics  to compare classification algorithms

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print('MLP Classifier Result:')
pred_a2 = grid_a2.predict(X_test)

acc2 = metrics.accuracy_score(Y_test,pred_a2)
print('Prediction Accuracy')
print(acc2)

facc2 = metrics.f1_score(Y_test,pred_a2,average= 'weighted')
print('Prediction F1 Score')
print(facc2)

print('Report')
print(classification_report(Y_test,pred_a2))

print('Confusion Matrix')
print(confusion_matrix(Y_test, pred_a2))


# In[67]:


#Displaying all metrics  to compare classification algorithms

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print('SGD Classifier Result:')
pred_a3 = grid_a3.predict(X_test)

acc3 = metrics.accuracy_score(Y_test,pred_a3)
print('Prediction Accuracy')
print(acc3)

facc3 = metrics.f1_score(Y_test,pred_a3,average= 'weighted')
print('Prediction F1 Score')
print(facc3)

print('Report')
print(classification_report(Y_test,pred_a3))

print('Confusion Matrix')
print(confusion_matrix(Y_test, pred_a3))


# In[68]:


#Plot graph final accuracy score

x_axis = np.arange(3)
y_axis = [facc1, facc2, facc3]

plt.bar(x_axis, y_axis, width=0.5)
plt.xticks(x_axis+0.5/10.,('GB','MLP','SGD'))
plt.ylabel('FT Prediction Accuracy')

plt.show()


# In[ ]:


#ROC curve
#ROC is a graphical plot which illustrates the performance of a binary 
#classifier system as its discrimination threshold is varied. AUC is the
#percentage of the ROC plot that is underneath the curve. AUC is useful as
#a single number summary of classifier performance.


# In[71]:


#store the predicted probablitites for Top 1 classifier
print('Probability mean for Classfier')
pred_best_prob = grid_a1.predict_proba(X_test)[:,1]
print pred_best_prob.mean()


# In[72]:


#AUC score
print('ROC AUC score')
best_auc = metrics.roc_auc_score(Y_test,pred_best_prob)
print(best_auc)


# In[74]:


#plot ROC curve

fpr, tpr, threshols = metrics.roc_curve(Y_test,pred_best_prob)
plt.plot(fpr, tpr)

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])

plt.title('ROC curve')

plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
plt.show()


# In[75]:


#Deriving the most important features affecting classification

# fit Random Forest model to the cross-validation data
from sklearn.ensemble import RandomForestClassifier

rand_forest = RandomForestClassifier()
rand_forest.fit(X_train, Y_train)
fimportances = rand_forest.feature_importances_

# make importance relative to the max importance
feature_importance = 100.0 * (fimportances / fimportances.max())
sorted_idex = np.argsort(feature_importance)
feature_names = list(X_test.columns.values)
feature_sorted = [feature_names[indice] for indice in sorted_idex]
pos = np.arange(sorted_idex.shape[0]) + .5
print 'Top 10 features are: '
for feature in feature_sorted[::-1][:10]:
    print feature

