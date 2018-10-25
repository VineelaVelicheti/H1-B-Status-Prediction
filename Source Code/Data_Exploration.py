
# coding: utf-8

# In[86]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[87]:


#importing data for year 2015
fifteen_df = pd.read_excel("/Users/vineevineela/Desktop/H-1B_FY2015.xlsx")


# In[88]:


#importing data for year 2016
sixteen_df = pd.read_excel("/Users/vineevineela/Desktop/H-1B_FY2016.xlsx")


# In[89]:


#importing data for year 2017
seventeen_df = pd.read_excel("/Users/vineevineela/Desktop/H-1B_FY2017.xlsx")


# In[90]:


#importing data for year 2018
eighteen_df = pd.read_excel("/Users/vineevineela/Desktop/H-1B_FY2018.xlsx")


# In[91]:


#create concise table to understand the data for year 2015

data_types = fifteen_df.dtypes.values
missing_count = fifteen_df.isnull().sum().values

unique_count = []
for attr in fifteen_df.columns:
    unique_count.append(fifteen_df[attr].unique().shape[0])

       
info ={'Attributes': fifteen_df.columns,
       'Attribute_Type': data_types,
       'MissingValue_Count':missing_count,
       'UniqueValue_Count': unique_count,     
       }
col_names = {'Attributes','Attribute_Type','MissingValue_Count','UniqueValue_Count'}

info_tab = pd.DataFrame(info,columns = col_names)
info_tab


# In[92]:


#create concise table to understand the data for year 2016

data_types = sixteen_df.dtypes.values
missing_count = sixteen_df.isnull().sum().values

unique_count = []
for attr in sixteen_df.columns:
    unique_count.append(sixteen_df[attr].unique().shape[0])

       
info ={'Attributes': sixteen_df.columns,
       'Attribute_Type': data_types,
       'MissingValue_Count':missing_count,
       'UniqueValue_Count': unique_count,      
       }
col_names = {'Attributes','Attribute_Type','MissingValue_Count','UniqueValue_Count'}

info_tab = pd.DataFrame(info,columns = col_names)
info_tab


# In[93]:


#create concise table to understand the data for year 2017

data_types = seventeen_df.dtypes.values
missing_count = seventeen_df.isnull().sum().values

unique_count = []
for attr in seventeen_df.columns:
    unique_count.append(seventeen_df[attr].unique().shape[0])

       
info ={'Attributes': seventeen_df.columns,
       'Attribute_Type': data_types,
       'MissingValue_Count':missing_count,
       'UniqueValue_Count': unique_count,      
       }
col_names = {'Attributes','Attribute_Type','MissingValue_Count','UniqueValue_Count'}

info_tab = pd.DataFrame(info,columns = col_names)
info_tab


# In[94]:


#create concise table to understand the data for year 2018

data_types = eighteen_df.dtypes.values
missing_count = eighteen_df.isnull().sum().values

unique_count = []
for attr in eighteen_df.columns:
    unique_count.append(eighteen_df[attr].unique().shape[0])

       
info ={'Attributes': eighteen_df.columns,
       'Attribute_Type': data_types,
       'MissingValue_Count':missing_count,
       'UniqueValue_Count': unique_count,      
       }
col_names = {'Attributes','Attribute_Type','MissingValue_Count','UniqueValue_Count'}

info_tab = pd.DataFrame(info,columns = col_names)
info_tab


# In[96]:


#For all common columns uniform names are given with respect to 2017 year dataset

#2015 year 
fifteen_df.rename(columns={'EMPLOYER_ADDRESS1':'EMPLOYER_ADDRESS'}, inplace=True)
fifteen_df.rename(columns={'NAIC_CODE':'NAICS_CODE'}, inplace=True)
fifteen_df.rename(columns={'TOTAL WORKERS':'TOTAL_WORKERS'}, inplace=True)
fifteen_df.rename(columns={'PW_WAGE_SOURCE':'PW_SOURCE'}, inplace=True)
fifteen_df.rename(columns={'PW_WAGE_SOURCE_YEAR':'PW_SOURCE_YEAR'}, inplace=True)
fifteen_df.rename(columns={'PW_WAGE_SOURCE_OTHER':'PW_SOURCE_OTHER'}, inplace=True)
fifteen_df.rename(columns={'WAGE_RATE_OF_PAY':'WAGE_RATE_OF_PAY_FROM'}, inplace=True)
fifteen_df.rename(columns={'H-1B_DEPENDENT':'H1B_DEPENDENT'}, inplace=True)
fifteen_df.rename(columns={'WILLFUL VIOLATOR':'WILLFUL_VIOLATOR'}, inplace=True)

#2016 year
sixteen_df.rename(columns={'H-1B_DEPENDENT':'H1B_DEPENDENT'}, inplace=True)
sixteen_df.rename(columns={'NAIC_CODE':'NAICS_CODE'}, inplace=True)
sixteen_df.rename(columns={'PW_WAGE_SOURCE':'PW_SOURCE'}, inplace=True)

#2018 year 
eighteen_df.rename(columns={'H-1B_DEPENDENT':'H1B_DEPENDENT'}, inplace=True)


# In[97]:


#converting common columns to same dtype and format

#2015 year 
fifteen_df['WAGE_RATE_OF_PAY_FROM'] = fifteen_df.WAGE_RATE_OF_PAY_FROM.apply(lambda a: a.split("-")[0])
fifteen_df['WAGE_RATE_OF_PAY_FROM'] = pd.to_numeric(fifteen_df['WAGE_RATE_OF_PAY_FROM'], errors= 'ignore')


# In[98]:


full_data = pd.concat([fifteen_df,sixteen_df,seventeen_df,eighteen_df],axis=0,ignore_index = True )


# In[99]:


full_data.shape


# In[100]:


#getting information about the target variable

#filling nan with category "blank" for bar graph representation  
full_data['CASE_STATUS'] = full_data['CASE_STATUS'].fillna('blank')

from collections import Counter
status_count = Counter(full_data['CASE_STATUS'])

x = status_count.keys()
y = status_count.values()

plt.bar(x,y, align='center', alpha=0.5, color='green')

plt.xlabel('Categories')
plt.ylabel('Number')

plt.title('CASE_STATUS')
 
plt.show()


# In[101]:


#filtered all rows with WITHDRAWN categories
full_data = full_data[(full_data[['CASE_STATUS']] != 'WITHDRAWN').all(axis=1)]
full_data = full_data[(full_data[['CASE_STATUS']] != 'CERTIFIED-WITHDRAWN').all(axis=1)]
full_data['CASE_STATUS'].value_counts()


# In[102]:


#displaying CERTIFIED and DENIED categories
from collections import Counter
status_count = Counter(full_data['CASE_STATUS'])

x = status_count.keys()
y = status_count.values()

plt.bar(x,y,alpha=0.5, color='yellow')

plt.xlabel('Categories')
plt.ylabel('Number')

plt.title('CASE_STATUS')
 
plt.show()


# In[103]:


#statistics for missing data in each column
null_count = full_data.isnull().sum().sort_values(ascending=False)
null_percent = (full_data.isnull().sum()/full_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([null_count, null_percent], axis=1, keys=['NULL_COUNT', 'NULL_PERCENT'])
missing_data


# In[104]:


#Dropping columns with NULL_PERCENT > 50% and columns that doesn't effect prediction.
#33 columns deleted from 53 columns
full_data = full_data.drop(['CASE_NUMBER','CASE_SUBMITTED','DECISION_DATE','EMPLOYER_BUSINESS_DBA','EMPLOYER_ADDRESS',
                            'EMPLOYER_POSTAL_CODE','EMPLOYER_PROVINCE','EMPLOYER_PHONE','EMPLOYER_PHONE_EXT',
                            'AGENT_ATTORNEY_CITY','AGENT_ATTORNEY_STATE','JOB_TITLE','SOC_NAME','NAICS_CODE',
                            'PW_WAGE_LEVEL','PW_SOURCE_YEAR','PW_SOURCE_OTHER','PUBLIC_DISCLOSURE_LOCATION',
                            'WORKSITE_COUNTY','WORKSITE_POSTAL_CODE','NEW_EMPLOYMENT','CONTINUED_EMPLOYMENT',
                            'CHANGE_PREVIOUS_EMPLOYMENT','NEW_CONCURRENT_EMPLOYMENT','CHANGE_EMPLOYER',
                            'FULL_TIME_POSITION','EMPLOYER_CITY','SUPPORT_H1B','EMPLOYER_ADDRESS2',
                            'AMENDED_PETITION','ORIGINAL_CERT_DATE','LABOR_CON_AGREE','WORKSITE_CITY'],1)


# In[105]:


#converting the combined data  from a dataframe to a CSV
full_data.to_csv('/Users/vineevineela/Desktop/H-1B_data.csv', encoding = 'utf-8',index = False)


# In[106]:


#converting the combined data  from a dataframe to a Excel
full_data.to_excel('/Users/vineevineela/Desktop/H-1B_data.xlsx',index = False)

