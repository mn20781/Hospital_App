#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#load dataframe with both sheets
#df_0 = pd.read_excel('/Users/michaelnaylor/Downloads/PatientData_ProgrammingAssignment.xlsx', 
                  # true_values = None, keep_default_na = True, na_filter=False, 
                 #  verbose=True, header = 1
                  # ,names=None, index_col=0, 
                 #  sheet_name = None, #sheet no in excel doc, none specifies all 
                 #  skiprows=2)
#df_0


# In[5]:


import pip
pip.main(["install", "openpyxl"])


# In[6]:


import pandas as pd
import numpy as np 


# In[10]:


url = 'https://raw.githubusercontent.com/mn20781/Hospital_App/main/PatientData_ProgrammingAssignment.xlsx'
import requests
from pprint import pprint

response = requests.get(url)
pprint(response.content)


# In[18]:


#load multiple excelsheets using ExcelFile()
pd.set_option('display.max_rows', None)
#xlsx = pd.ExcelFile('/Users/michaelnaylor/Downloads/PatientData_ProgrammingAssignment.xlsx')
#load the first sheet
url = 'https://raw.githubusercontent.com/mn20781/Hospital_App/main/PatientData_ProgrammingAssignment.xlsx'

df1 = pd.read_excel(url, "Sheet1", true_values = None, keep_default_na = True, na_filter=False, #detect
                   #missing values
                   verbose=True, header = 1
                   ,names=None, index_col=None, 
                    #sheet no in excel doc, none specifies all 
                   skiprows=2)
#load the second sheet |
df2 = pd.read_excel(url, "Sheet2", true_values = None, keep_default_na = True, na_filter=False, 
                   verbose=True, header = 1
                   ,names=None, index_col=None, 
                   #sheet_name = 1,
                     skiprows=2)


# In[19]:


df1.info() #check the dataframe


# In[20]:


#rename the columns using a dictionary 
col_map_df = {
    "Unnamed: 0" : "",
    "Patient_ID" : "patient_id",
    "ER_positive" : "er_positive",
    "PR_positive" : "pr_positive",
    "HER2_positive" : "her2_positive",
    "Unnamed: 6" : "",
    "Field Name" : "field_name",
    "Description" : "description"
}


# In[21]:


#df.iloc["Patient_ID"]


# In[22]:


patient_results = df1.rename(columns=col_map_df) #rename the dataframes columns
                                                #place in new variable patient_results


# In[23]:


patient_results.columns #look up the column labels of the dataframe


# In[25]:


patient_results.head(n=5) #check the first 5 results in patient_results


# In[26]:


#patient_results['description'] #check the description of the data 


# In[27]:


#patient_results.describe() #return the summary statistics and a few quartiles


# In[28]:


#patient_results.dtypes.value_counts(sort=True)


# In[29]:


#patient_results.info() #prints datatype information in addition to the count of non-null values
          #also list amount of memory used by the dataframe


# In[30]:


patient_results.shape #shows how many rows and columns are in the dataframe 


# In[33]:


df2.info() #same as above


# In[36]:


#renaming columns for treatment information 
col_map = {
    "Unnamed: 0" : "",
    "Patient_ID" : "patient_id",
    "Drug_admin_date" : "drug_admin_date",
    "Length" : "length",
    "Field Name" : "field_name",
    "Description" : "description"
    
}


# In[37]:


patient_treatment = df2.rename(columns=col_map) #rename using the rename function
patient_treatment.head(n=5) #check the first 5 results


# In[38]:


dates = patient_treatment[['patient_id', 'drug_admin_date']] #dataframe of patient id and admindate


# In[48]:


dates.head(n=5) #first 5results of this dataframe


# In[47]:


#loop throught dtframe and see the items 
for patient_id, date in dates.items():
    print(f'Label: {patient_id}')
    print(f'Content: {date}', sep='\n')


# In[49]:


patient_treatment.info() #check the info
#patient_treatment.shape


# In[53]:


patient_treatment[["drug_admin_date"]].dtypes #check the data type of the drug_admin_date column 


# In[52]:


#patient_treatment["patient_id"].value_counts()


# In[57]:


type(patient_treatment[["patient_id"]]) #check what type of object patient_treatment['patient_id'] is


# In[58]:


### df_1["Drug_admin_date"].dtype #check the dytpe of 'Drug_admin_date'
#take user input and return patient information 
def cal_treatment(patient_id=0):
    user_input = int(input('Enter Patient ID :',)) #grab the user input 
    #when the user enters the patient id, the function should go through the database to 
    #find a similar value
    p_id = patient_treatment['patient_id'] == user_input 
    
    if p_id.any():
        #calculates the treatment length of the patient
        patient_treatment['length'] = patient_treatment["drug_admin_date"] - patient_treatment["drug_admin_date"].shift(1)
        
        print(patient_treatment.loc[p_id, ['patient_id', 'drug_admin_date', 'length']])
    else:
        print('Invalid Patient ID, Please Try again')
        cal_treatment()
        
    
   # result = patient_treatment["drug_admin_date"] - patient_treatment["drug_admin_date"]
    #if result > 0 :
      #  return
    #elif result <0:
        
      #  print(result, 'days of treatment')
cal_treatment() #calls the function 


# In[59]:


#print(patient_treatment.iloc[0:10], patient_results.iloc[0:10]) 
#take first 10 results and display them from both dataframes 


# In[61]:


print(patient_treatment.memory_usage()) #check the memory usage of the dataframe 


# In[64]:


#treatment_l = patient_treatment.copy()
#treatment_l.columns


# In[65]:


#treatment_l=patient_treatment.groupby(['patient_id','drug_admin_date'])['length'].sum()
#treatment_l.astype


# In[66]:


#data = df_1.sort_index()
#data_copy = data.copy()


# In[67]:


#data_copy[['Length']]


# In[68]:


#data_copy["Length"] = data_copy["Drug_admin_date"] - data_copy["Drug_admin_date"] #calculate
#calculate treatment length 


# In[69]:


#data_copy


# In[70]:


#building a multiindex from the column values
#patient_data = patient_treatment.set_index(['patient_id', 'drug_admin_date', 'length']) #indexing the columns 
#patient_data.head()


# In[71]:


#†rying to see what setting patient ID as the lone index will do 
#pdata=  data.set_index(['Patient_ID'])
#pdata.head()


# In[72]:


#need to write a function to take in the patient id number 
#and check the next ids to see if they are similar if not,
#then if it calculates the length of treatment from the 1st index with the id to the nth


# In[73]:


#pdd =  data.set_index(['Drug_admin_date'])
#pdd


# In[74]:


#length = pdd.iloc[0:]


# In[2]:


#pdata["Length"]= pdata['Drug_admin_date'] - pdata['Drug_admin_date'].shift(1)
#pdata


# In[75]:


#pdata.loc[2120]


# In[76]:


#patient_data_copy = patient_data.copy()


# In[77]:


#patient_data.loc[:,[2:]]


# In[79]:


#idx = pd.IndexSlice 
#df_1.loc[idx[:,"Patient_ID"], idx[:,'Drug_admin_date']]


# In[80]:



#df_1 = df_1.sort_values(by="Drug_admin_date") 


# In[81]:


#df_1['Length'] = pd.to_datetime(df_1['Length'])


# In[82]:


#for n in df_1["Patient_ID"]:
  #  if n == df_1["Patient_ID"]:
   #     df_1[["Length"]] = df_1[["Drug_admin_date"]] - df_1[["Drug_admin_date"]].shift(1)
  #  else:
   #     df_1["Length"] = 0


# In[86]:


patient_treatment[["length"]] = patient_treatment[["drug_admin_date"]] - patient_treatment[["drug_admin_date"]].shift(1)
patient_treatment


# In[ ]:





# In[46]:


df_1[["Length"]] = df_1.Drug_admin_date - df_1.Drug_admin_date.shift(1)
#df_1[["Length"]] = df_1.Drug_admin_date / np.timedelta64(1, 'W')
df_1


# In[1182]:


#df_1.loc[:,"Length" < pd.Timedelta(1,'D')] = 0
df_1.loc[df_1["Length"] < pd.Timedelta(1,'D')] = 0 


# In[1183]:


#df_1["Length"][df_1["Length"] < pd.Timedelta(1,'D')]
df_1


# In[87]:


#df_1.loc[2:9]


# In[88]:


#df_1.loc[df_1["Length"] <  pd.Timedelta(1,'D')] = 0


# In[89]:


#df_1


# In[90]:


#df_1[["Length"]] = df_1[["Drug_admin_date"]] - df_1[["Drug_admin_date"]].shift(1)
#df_1


# In[91]:


#type(treatment_length)


# In[1088]:


for n in treatment_length:
    if n < 0 :
        treatment_length[n] == 0


# In[92]:


#treatment_length = np.datetime64(drug_admin_date[:], 'W') - np.datetime64(drug_admin_date[:], 'W').shift[1]
#print("Patient",Patient_ID[0], "had treatment for",treatment_length, "Weeks")


# In[101]:


#df['ER_positive'].dtype #check the data type of ER_positive column 
patient_results.shape


# In[96]:


patients = patient_results.groupby(['er_positive']) #group ER patients in a container called patients
patients.size()
#patients.first()
#patients.reset_index()
#positive_ER = patients.get_group(1) #put the group of positive patients in the variable positive_ER
#positive_ER


# In[12]:


pizza = [n for n in df_1["Patient_ID"] if n == positive["Patient_ID"]]
pizza


# In[13]:


df_1


# In[17]:


positive


# In[1218]:


positive.columns


# In[1223]:


positive[positive["Patient_ID"].isin(df_1["Patient_ID"])]


# In[62]:



#df_1["treatment_length"] = 
treatment_length = df_1.loc[["Patient_ID"] == positive.loc["Patient_ID"]]
print(df_1)


# In[896]:


for n in df:
    if df[["Patient_ID"]].iloc[n] == patients.iloc[n]:
        print(df[["Patient_ID"]])
    


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[418]:


df_1.columns


# In[80]:


df_1.loc[50]


# In[78]:


df1_index = pd.DataFrame(df_1, 
                        index=[["Patient_ID"]], 
                        columns=(["Patient_ID", "Drug_admin_date"]))
df1_index
df1_index_copy = df1_index.copy()

df1_index


# In[71]:


df_1.columns


# In[70]:


type(df_1)


# In[74]:


type(df1_index)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[585]:


for i in df1_index.items():
    if i  == df1_index:
        print(i)


# In[558]:


row = next(df.iterrows())[1]
row
df1_index.keys


# In[703]:


index = pd.Index(df1_index, name = "Patient_Info")

#deef =  pd.DataFrame(index, 
                      #  index=index,
                     #   columns=(["Patient_ID", "Drug_admin_date"]))
#deef
index


# In[1210]:


idd_ =df_1.loc['Patient_ID'] == df['Patient_ID']


# In[451]:


print(len(df))
df.columns


# In[509]:


#check the shape of the variable
#df_index.shape



# In[512]:


#creating an index of the data frame that we can use to sort through the data using Pandas Dataframe function
df_index = pd.DataFrame(df, 
                        index=range(0,37), 
                        columns=(["Patient_ID", "drug_390_admin_flag",
                                  "ER_positive","PR_positive", "HER2_positive"]))
df_index
#it needs to search row by row, at the coordinates to find the value and use that to search the coordinates of 
#next columns


# In[770]:


#df_index.query( "drug_390_admin_flag"and"PR_positive")
#df_index[( df_index["ER_positive"]< df_index["PR_positive"])]
#df_index["drug_390_admin_flag"]
#df_index[df_index["Patient_ID"].isin([df_index["PR_positive"]])]


# In[954]:


df[["PR_positive"]], df[["Patient_ID"]]


# In[947]:


patients = df.groupby('PR_positive')
#print(df.groupby(['Patient_ID']).groups)
patients.head()


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[458]:


#grab the index of the column 
#test example
df_index.iloc[36]


# In[ ]:





# In[958]:


df_index.iloc[0:], df.columns.get_indexer(['Patient_ID', 
                                     "PR_positive", "ER_positive"])


# In[880]:


df_index.iloc[0:], df.columns.get_indexer(['Patient_ID']) 


# In[1059]:


df.loc[[1],["Patient_ID"]]


# In[1016]:


positive_tests.columns


# In[1227]:


tests = positive_tests[positive_tests["Patient_ID"].isin(df_1["Patient_ID"])]


# In[1248]:


#positive_tests["Patient_ID"].isin(df_1["Patient_ID"])
pos = positive_tests["Patient_ID"].items
pos


# In[1239]:


df_1["Positives"] = positive_tests["Patient_ID"] for n in positive_tests["Patient_ID"]


# In[1243]:


df_1


# In[1298]:


#df_1[["Positives"]]= df_1.loc[df_1["Patient_ID"] & positive_tests["Patient_ID"], 
                            #  df_1["Positive"]] = 0
posit=  positive_tests["Patient_ID"]
#select items in patient ID
df_1.groupby([["Patient_ID"], posit])


# In[1251]:


#df_1[df_1['Patient_ID'] == positive_tests[['Patient_ID']].items]
df_1['Patient_ID']


# In[879]:


df.iloc[[22], df.columns.get_indexer(['Patient_ID',  
                                     "PR_positive", "ER_positive"])], df.iloc[[0], df.columns.get_indexer(['Patient_ID',  
                                     "PR_positive", "ER_positive"])]


# In[523]:


df.values


# In[486]:


#criterion = df['Patient_ID'].map(lambda x: x.startswith(1))
positive_tests = {'drug_390_admin_flag': [1],"PR_positive":[1], "ER_positive":[1], "HER2_positive":[1] }
#s= df.isin(positive_tests).any(1)
df.query('a < b & b < c')


# In[407]:


#data frame with data from the first sheet
#df = pd.read_excel('/Users/michaelnaylor/Downloads/PatientData_ProgrammingAssignment.xlsx', 
                 #  true_values = None, keep_default_na = True, na_filter=False, 
                 #  verbose=True, header = 1
                 #  ,names=None, index_col=None, 
                 #  sheet_name = 0, #sheet no in excel doc, 
                 #  skiprows=2)

                  
#df.head() #print the first 5 values to see if data has loaded properly 
#df.dropna() #drop the NaN values in the dataset 
#df.dropna(axis='columns')
#df.head() #shows the first 5 elements
#


# In[285]:


#drug_data = df.values
#drug_data


# In[393]:


df_Z = pd.DataFrame(
                      index=list('012345'))#,
                      #columns=list('Patient_ID','ER_positive'))
df_Z


# In[350]:


#drug_data[[
type(df)


# In[361]:


#df[["Patient_ID"]]
patient_ID = df[["Patient_ID"]].values
#print(df.columns) #show the column names 
#df.columns[2] #show the label of column in index 2
print(len(patient_ID))
print(patient_ID[36], patient_ID[4])

#df[["Patient_ID"]].head()
hist = df[["drug_390_admin_flag"]].hist(grid=True, bins=20)#creating a histogram to visualise the data 
hist = df[["ER_positive"]].hist(grid=True, bins = 20)
df[["ER_positive"]]


# In[363]:


for n in df[["ER_positive"]]:
    if n == 1:
        print (n)


# In[613]:


drug_admin= df.loc[df["drug_390_admin_flag"] >= 1]
len(drug_admin)


# In[18]:


er_Positive = df.loc[df["ER_positive"] >= 1] #print rows with column values 1 or greater
len(er_Positive)


# In[19]:


pr_Positive =df.loc[df["PR_positive"] >= 1]
len(pr_Positive)


# In[33]:


df.count()


# In[813]:


df_1["Patient_ID"].count()


# In[925]:


pr_Positive.to_numpy()


# In[924]:


pr_Positive["Patient_ID"].isin(df_1["Patient_ID"]) #check if the values in pr_positive are in df_1


# In[909]:


positive_IDs = pr_Positive.groupby('Patient_ID')
for n, name in positive_IDs:
    print(n)
   # if n == df_1[["Patient_ID"]]:
      #  print(df_1["drug_admin_date"])


# In[966]:


positive_IDs.dtype


# In[960]:


positive_tests


# In[610]:


er_Negative = df.loc[df["ER_positive"] == 0]
len(er_Negative)


# In[615]:


pr_Negative = df.loc[df["PR_positive"] == 0]
len(pr_Negative)


# In[961]:


positive_tests = [pr_Positive, er_Positive] #creates a list
len(positive_tests)


# In[965]:


positive_tests.i


# In[20]:


positive_tests = df.loc[(df["PR_positive"] == 1) | (df["ER_positive"] ==1)]


# In[21]:


positive_tests


# In[139]:


patient_data_copy.loc[, Drug_admin_date]- patient_data_copy.loc[, :].shift(1)


# In[140]:


patient_data_copy.xs(level='Patient_ID')


# In[143]:


patient_data_copy.iloc[1:,]


# In[126]:


pizza = [n for n in positive_tests["Patient_ID"] for y in df_1["Patient_ID"]if n == y]
print(len(pizza), pizza)


# In[47]:


df_1copy = df_1.copy()
df_1copy


# In[ ]:


df_1["Positives"] = positive_tests["Patient_ID"] for n in positive_tests["Patient_ID"]


# In[52]:


df_1copy["length"] = #[df_1['Patient_ID'].duplicated() == True for n in positive_tests["Patient_ID"] 
                      for y in df_1["Patient_ID"]if n == y]= df_1[["Drug_admin_date"]] - df_1[["Drug_admin_date"]].shift(1)


# In[59]:


df_1copy["length"] = ['Positive'
                      for n in positive_tests["Patient_ID"]
                      for y in df_1["Patient_ID"]
                      if n == y]#df_1[["Drug_admin_date"]] - df_1[["Drug_admin_date"]].shift(1)


# In[41]:





# In[32]:


df_1["Patient_ID"].value_counts()


# In[33]:


len(df_1["Patient_ID"])


# In[37]:


df_1.loc[df_1["Patient_ID"] == 9489]


# In[29]:


df_1["Positive"] = [n for n in positive_tests["Patient_ID"] for y in df_1["Patient_ID"]if n == y]


# In[1093]:


positive_tests.columns


# In[937]:


pr_Negative["Patient_ID"].isin(df_1["Patient_ID"])


# In[1]:


negative_tests = df.loc[(df["PR_positive"] == 0) & (df["ER_positive"] ==0)]


# In[854]:


#positive_tests.values
df['Patient_ID'].values


# In[1108]:


len(df)-len(positive_tests) 


# In[1106]:


#loop through the patients with positive tests
for n in positive_tests["Patient_ID"]:
   # for ide in df_1[['Patient_ID']]:
       # while df[n] == df_1[ide]:
            print("Patient_ID:",n)
print("There are:", len(positive_tests), "positive tests")


# In[1104]:


#create class to sort patient information  
class Patient_Info:
    def __init__(self, patient_ID, diagnosis, treatment_length):
        self.patient_ID = patient_ID
        self.diagnosis = diagnosis
        self.treatment_length = treatment_length


# In[972]:





# In[781]:


negative_tests


# In[799]:


grp1 = negative_tests.groupby('Patient_ID')
for n, name in grp1:
    print(n)
   # print(name)

   # print()


# In[773]:


positive_tests_0 = df.loc[df["Patient_ID"].isin(positive_tests)]
positive_tests_0


# In[776]:


df.loc[:"Patient_ID"]


# In[833]:


#creating a function to loop through the values of df["Patient_ID"]
#def treatment_length():
for n in df["Patient_ID"].values:
    for num in range(len(positive_tests)):
        if positive_tests[num] == positive_tests:
            print(positive_tests[num])
            Q
        
                


# In[823]:


positive_tests


# In[641]:


positive_tests[0]


# In[651]:


df.loc[["Patient_ID"]]


# In[638]:


total_positives = len(positive_tests[0]) + len(positive_tests[1])
total_positives


# In[317]:


#need to create a class that allows you to store the patient ID values as the value to be called. 
#The function should give you the ability to search for an ID by imputting a list of values
#further consideration should be kept for interaction, in the UI such as a searchbar that attempts to predict
#the number that will be entered. The function can only accept INTs that are 4 numbers long. 
type(df[['Patient_ID']])


# In[192]:


df_1 = pd.read_excel('/Users/michaelnaylor/Downloads/PatientData_ProgrammingAssignment.xlsx', 
                   true_values = None, keep_default_na = True, na_filter=False, 
                   verbose=True, header = 1
                   ,names=None, index_col=None, 
                   sheet_name = 1, skiprows=2)
df_1.head()


# In[210]:


df_1[['Drug_admin_date']], df_1[['Patient_ID']]


# In[67]:



df_1copy


# In[68]:


features = ["Patient_ID", "Drug_admin_date"]
print(df_1[features])


# In[ ]:


#for ID in features:


# In[ ]:


#function to calculate the length of a patients treatment

#def treatmenttime():
    
    


# In[211]:


df['drug_390_admin_flag'].values
print("Column Name:",df.columns[2],"\nValues:",df['drug_390_admin_flag'].values,
      "\nNo. of Values:",len(df['drug_390_admin_flag'])
      ,"\nMean:", sum(df["drug_390_admin_flag"])/len(df["drug_390_admin_flag"]))


# In[ ]:


df


# In[217]:


data = df.iloc[:,2].values
data


# In[235]:


print("First 5 numbers in this ndarray:",data[0:5] )#data[1])


# In[241]:


#we need a function that compares the key to the value of the vectors in each column 
patient_ID = df_1['Patient_ID'].values
drug_admin_date = df_1['Drug_admin_date'].values
patient_ID


# In[268]:


#df_1[Patient_ID]
#create a function which counts how many times an ID appears in the array, 
#then arranges the  patient ID in sequential order, smallest to largest without repetitions
patient_ID[0]

#for x, ID in np.ndenumerate(patient_ID):
    #print(x, ID)
    


# In[239]:


#test out the patient ID array 
patient_ID[0]


# In[242]:


#test out drug admin date data
drug_admin_date[0]


# In[256]:


#variable to store the treatment length
treatment_length = np.datetime64(drug_admin_date[0], 'W') - np.datetime64(drug_admin_date[1], 'W') 
print("Patient",Patient_ID[0], "had treatment for",treatment_length, "Weeks")


# In[214]:


print("This is a",type(data),"\nShowing a 1D array:", data, "\nThere are",len(data),"examples in this dataset")


# In[9]:


data = df.iloc[:,1].values


# In[10]:


print(type(data), data)


# In[11]:


data = df.iloc[:,3].values


# In[12]:


print(type(data), data)


# In[13]:


data = df.iloc[:,4].values
print(type(data), data)


# In[145]:


#print(df.describe)


# In[147]:


df.dropna()


# In[143]:


print(df.describe)


# In[198]:


print(df['Patient_ID'], df['drug_390_admin_flag'], df['ER_positive'])
#df


# In[27]:


old=np.array([0,1,1,1,0,1,1,0,0,1,0])


# In[46]:


old_mean=(sum(old)/len(old))#get the average of the old data 


# In[44]:


print(old_mean)


# In[32]:





# In[ ]:




