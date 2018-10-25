
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import Counter
import time


# In[2]:


data = pd.read_csv("/Users/vineevineela/Desktop/H-1B_data.csv")


# In[3]:


data.dtypes


# In[4]:


data.shape


# In[5]:


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


# In[22]:


data.dtypes


# In[ ]:


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


# In[23]:


#delete column EMPLOYER_NAME
data = data.drop(['EMPLOYER_NAME'],axis=1)


# In[24]:


data = data.dropna()


# In[25]:


from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)

data['INCREMENT'] = lb.fit_transform(data["INCREMENT"])
data['CASE_STATUS'] = lb.fit_transform(data["CASE_STATUS"])


# In[26]:


data.dtypes


# In[28]:


data.shape


# In[27]:


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


#X_train.select_dtypes(include=['object'])
#Feauture Selection : Select KBest
import time
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

start_time = time.clock()
best_selector = SelectKBest(f_classif, k=100)
best_selector = best_selector.fit(train_data,Y_train)

Train_PCA = best_selector.transform(train_data)
print(Train_PCA.shape)

Test_PCA = best_selector.transform(test_data)
print(Test_PCA.shape)

print("Time Taken in seconds:", time.clock() - start_time) 


# In[32]:


#decision tree

from sklearn import tree
from sklearn.metrics import accuracy_score

start_time = time.clock()
c1 = tree.DecisionTreeClassifier()
c1.fit(Train_PCA,Y_train)
Dectree_Y =c1.predict(Test_PCA)

from sklearn.metrics import accuracy_score
Dec_Accuracy = accuracy_score(Y_test, Dectree_Y)
print("Decision Tree Accuracy = ",Dec_Accuracy)

from sklearn.metrics import f1_score
Dec_F1 = f1_score(Y_test,Dectree_Y, average= 'weighted')
print("Decision Tree F1Score = ",Dec_F1)

print("Time Taken in seconds:", time.clock() - start_time) 


# In[34]:


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


# In[40]:


#Bagging

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

start_time = time.clock()
c3 = BaggingClassifier(n_estimators=500)
c3.fit(Train_PCA,Y_train)
Bag_Y =c3.predict(Test_PCA)

from sklearn.metrics import accuracy_score
Bag_Accuracy = accuracy_score(Y_test, Bag_Y)
print("Bagging classifier accuracy = ",Bag_Accuracy)

from sklearn.metrics import f1_score
Bag_F1 = f1_score(Y_test,Bag_Y,average= 'weighted')
print("Bagging classifier F1Score = ",Bag_F1)

print("Time Taken in seconds:", time.clock() - start_time) 


# In[35]:


#Classification algorithm KNN

from sklearn.neighbors import KNeighborsClassifier

start_time = time.clock()
neigh = KNeighborsClassifier(n_neighbors=1000, weights='uniform', algorithm='auto')
neigh.fit(Train_PCA,Y_train)
KNN_y = neigh.predict(Test_PCA)

#Accuracy calculation
from sklearn.metrics import accuracy_score
KNN_Accuracy = accuracy_score(Y_test, KNN_y)
print("KNN Accuracy = ",KNN_Accuracy)

from sklearn.metrics import f1_score
KNN_F1 = f1_score(Y_test,KNN_y,average= 'weighted')
print("KNN F1Score = ",KNN_F1)

print("Time Taken in seconds:", time.clock() - start_time) 


# In[36]:


#Classication algorithm : Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

start_time = time.clock()
RF= RandomForestClassifier(n_estimators=500,n_jobs=-1,criterion='gini',class_weight = 'balanced')
RF.fit(Train_PCA,Y_train)
RF_Y = RF.predict(Test_PCA)

#Accuracy calculation
from sklearn.metrics import accuracy_score
RF_Accuracy = accuracy_score(Y_test, RF_Y)
print("Random forest Accuracy = ",RF_Accuracy)

from sklearn.metrics import f1_score
RF_F1 = f1_score(Y_test,RF_Y,average= 'weighted')
print("Random forest F1 Score = ",RF_F1)

print("Time Taken in seconds:", time.clock() - start_time) 


# In[37]:


# Classification algorithm : Adaboost

from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,ExtraTreesClassifier

start_time = time.clock()
AdaB_model=AdaBoostClassifier(RandomForestClassifier(n_estimators=500,n_jobs=-1,criterion='gini',class_weight = 'balanced'))
AdaB_model.fit(Train_PCA,Y_train)
AdaB_Y = AdaB_model.predict(Test_PCA)

#Accuracy calculation
from sklearn.metrics import accuracy_score
AdaB_Accuracy = accuracy_score(Y_test, AdaB_Y)
print("Adaboost accuracy =",AdaB_Accuracy)

from sklearn.metrics import f1_score
AdaB_F1 = f1_score(Y_test,AdaB_Y,average= 'weighted')
print("Adaboost F1Score =",AdaB_F1)

print("Time Taken in seconds:", time.clock() - start_time) 


# In[41]:


#CLASSIFICATION
#Extra Tress Classifier

from sklearn.ensemble import ExtraTreesClassifier

start_time = time.clock()
extra_trees = ExtraTreesClassifier(n_estimators=2000,n_jobs=-1,criterion='gini',class_weight = 'balanced')
ext_model = extra_trees.fit(Train_PCA,Y_train)
pred_et = ext_model.predict(Test_PCA)

from sklearn.metrics import accuracy_score
EXT_Accuracy = accuracy_score(Y_test, pred_et)
print("Extra Trees Accuracy=",EXT_Accuracy)

from sklearn.metrics import f1_score
EXT_F1 = f1_score(Y_test, pred_et, average= 'weighted') 
print("Extra Trees F1Score=",EXT_F1)

print("Time Taken in seconds:", time.clock() - start_time) 


# In[38]:


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


# In[39]:


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


# In[ ]:


#plotting a graph comparing various Accuracies

x_axis = np.arange(9)
y_axis = [Dec_Accuracy , Grad_Accuracy, Bag_Accuracy, 
          KNN_Accuracy, RF_Accuracy, AdaB_Accuracy, EXT_Accuracy, MLP_Accuracy,  SGD_Accuracy]

pt.bar(x_axis, y_axis, width=0.5)
pt.xticks(x_axis+0.5/10.,('DecTree','GrdBst',
           'Bag','KNN','RF',
            'AdaBst','ExtTree','MLP','SGD'))
pt.ylabel('Accuracy')

pt.show()


# In[ ]:


#plotting a graph comparing various F1-Scores

x_axis = np.arange(9)
y_axis = [Dec_F1 , Grad_F1, Bag_F1, 
          KNN_F1, RF_F1, AdaB_F1, EXT_F1, MLP_F1,  SGD_F1]

pt.bar(x_axis, y_axis, width=0.5)
pt.xticks(x_axis+0.5/10.,('DecTree','GrdBst',
           'Bag','KNN','RF',
            'AdaBst','ExtTree','MLP','SGD'))
pt.ylabel('Accuracy')

pt.show()

