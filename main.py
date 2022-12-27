#%matplotlib inline


# load dataframe with both sheets
# df_0 = pd.read_excel('/Users/michaelnaylor/Downloads/PatientData_ProgrammingAssignment.xlsx',
# true_values = None, keep_default_na = True, na_filter=False,
#  verbose=True, header = 1
# ,names=None, index_col=0,
#  sheet_name = None, #sheet no in excel doc, none specifies all
#  skiprows=2)
# df_0

import pip

pip.main(["install", "openpyxl"])

import pandas as pd
import numpy as np

# load multiple excelsheets using ExcelFile()
pd.set_option('display.max_rows', None)
xlsx = pd.ExcelFile('/Users/michaelnaylor/Downloads/PatientData_ProgrammingAssignment.xlsx')
# load the first sheet
df = pd.read_excel(xlsx, "Sheet1", true_values=None, keep_default_na=True, na_filter=False,  # detect
                   # missing values
                   verbose=True, header=1
                   , names=None, index_col=None,
                   # sheet no in excel doc, none specifies all
                   skiprows=2)
# load the second sheet |
df_1 = pd.read_excel(xlsx, "Sheet2", true_values=None, keep_default_na=True, na_filter=False,
                     verbose=True, header=1
                     , names=None, index_col=None,
                     # sheet_name = 1,
                     skiprows=2)

df.info()  # check the dataframe

# rename the columns using a dictionary
col_map_df = {
    "Unnamed: 0": "",
    "Patient_ID": "patient_id",
    "ER_positive": "er_positive",
    "PR_positive": "pr_positive",
    "HER2_positive": "her2_positive",
    "Unnamed: 6": "",
    "Field Name": "field_name",
    "Description": "description"
}

# df.iloc["Patient_ID"]

patient_results = df.rename(columns=col_map_df)  # rename the dataframes columns
# place in new variable patient_results

patient_results.columns  # look up the column labels of the dataframe

patient_results.head(n=5)  # check the first 5 results in patient_results

# patient_results['description'] #check the description of the data

# patient_results.describe() #return the summary statistics and a few quartiles

# patient_results.dtypes.value_counts(sort=True)

# patient_results.info() #prints datatype information in addition to the count of non-null values
# also list amount of memory used by the dataframe

patient_results.shape  # shows how many rows and columns are in the dataframe

df_1.info()  # same as above

# renaming columns for treatment information
col_map = {
    "Unnamed: 0": "",
    "Patient_ID": "patient_id",
    "Drug_admin_date": "drug_admin_date",
    "Length": "length",
    "Field Name": "field_name",
    "Description": "description"

}

patient_treatment = df_1.rename(columns=col_map)  # rename using the rename function
patient_treatment.head(n=5)  # check the first 5 results

dates = patient_treatment[['patient_id', 'drug_admin_date']]  # dataframe of patient id and admindate

dates.head(n=5)  # first 5results of this dataframe

# loop throught dtframe and see the items
for patient_id, date in dates.items():
    print(f'Label: {patient_id}')
    print(f'Content: {date}', sep='\n')

patient_treatment.info()  # check the info
# patient_treatment.shape

patient_treatment[["drug_admin_date"]].dtypes  # check the data type of the drug_admin_date column

# patient_treatment["patient_id"].value_counts()

type(patient_treatment[["patient_id"]])  # check what type of object patient_treatment['patient_id'] is


### df_1["Drug_admin_date"].dtype #check the dytpe of 'Drug_admin_date'
# take user input and return patient information
def cal_treatment(patient_id=0):
    user_input = int(input('Enter Patient ID :', ))  # grab the user input
    # when the user enters the patient id, the function should go through the database to
    # find a similar value
    p_id = patient_treatment['patient_id'] == user_input

    if p_id.any():
        # calculates the treatment length of the patient
        patient_treatment['length'] = patient_treatment["drug_admin_date"] - patient_treatment["drug_admin_date"].shift(
            1)

        print(patient_treatment.loc[p_id, ['patient_id', 'drug_admin_date', 'length']])
    else:
        print('Invalid Patient ID, Please Try again')
        cal_treatment()


# result = patient_treatment["drug_admin_date"] - patient_treatment["drug_admin_date"]
# if result > 0 :
#  return
# elif result <0:

#  print(result, 'days of treatment')
cal_treatment()  # calls the function

# print(patient_treatment.iloc[0:10], patient_results.iloc[0:10])
# take first 10 results and display them from both dataframes

print(patient_treatment.memory_usage())  # check the memory usage of the dataframe

# treatment_l = patient_treatment.copy()
# treatment_l.columns

# treatment_l=patient_treatment.groupby(['patient_id','drug_admin_date'])['length'].sum()
# treatment_l.astype

# data = df_1.sort_index()
# data_copy = data.copy()

# data_copy[['Length']]

# data_copy["Length"] = data_copy["Drug_admin_date"] - data_copy["Drug_admin_date"] #calculate
# calculate treatment length

# data_copy

# building a multiindex from the column values
# patient_data = patient_treatment.set_index(['patient_id', 'drug_admin_date', 'length']) #indexing the columns
# patient_data.head()

# †rying to see what setting patient ID as the lone index will do
# pdata=  data.set_index(['Patient_ID'])
# pdata.head()

# need to write a function to take in the patient id number
# and check the next ids to see if they are similar if not,
# then if it calculates the length of treatment from the 1st index with the id to the nth

# pdd =  data.set_index(['Drug_admin_date'])
# pdd

# length = pdd.iloc[0:]

# pdata["Length"]= pdata['Drug_admin_date'] - pdata['Drug_admin_date'].shift(1)
# pdata

# pdata.loc[2120]

# patient_data_copy = patient_data.copy()

# patient_data.loc[:,[2:]]

# idx = pd.IndexSlice
# df_1.loc[idx[:,"Patient_ID"], idx[:,'Drug_admin_date']]


# df_1 = df_1.sort_values(by="Drug_admin_date")

# df_1['Length'] = pd.to_datetime(df_1['Length'])

# for n in df_1["Patient_ID"]:
#  if n == df_1["Patient_ID"]:
#     df_1[["Length"]] = df_1[["Drug_admin_date"]] - df_1[["Drug_admin_date"]].shift(1)
#  else:
#     df_1["Length"] = 0

patient_treatment[["length"]] = patient_treatment[["drug_admin_date"]] - patient_treatment[["drug_admin_date"]].shift(1)
patient_treatment

df_1[["Length"]] = df_1.Drug_admin_date - df_1.Drug_admin_date.shift(1)
# df_1[["Length"]] = df_1.Drug_admin_date / np.timedelta64(1, 'W')
df_1

# df_1.loc[:,"Length" < pd.Timedelta(1,'D')] = 0
df_1.loc[df_1["Length"] < pd.Timedelta(1, 'D')] = 0

# df_1["Length"][df_1["Length"] < pd.Timedelta(1,'D')]
df_1

# df_1.loc[2:9]

# df_1.loc[df_1["Length"] <  pd.Timedelta(1,'D')] = 0


# df_1

# df_1[["Length"]] = df_1[["Drug_admin_date"]] - df_1[["Drug_admin_date"]].shift(1)
# df_1

# type(treatment_length)

for n in treatment_length:
    if n < 0:
        treatment_length[n] == 0

# treatment_length = np.datetime64(drug_admin_date[:], 'W') - np.datetime64(drug_admin_date[:], 'W').shift[1]
# print("Patient",Patient_ID[0], "had treatment for",treatment_length, "Weeks")


# df['ER_positive'].dtype #check the data type of ER_positive column
patient_results.shape

patients = patient_results.groupby(['er_positive'])  # group ER patients in a container called patients
patients.size()
# patients.first()
# patients.reset_index()
# positive_ER = patients.get_group(1) #put the group of positive patients in the variable positive_ER
# positive_ER

pizza = [n for n in df_1["Patient_ID"] if n == positive["Patient_ID"]]
pizza

df_1

positive

positive.columns

positive[positive["Patient_ID"].isin(df_1["Patient_ID"])]

# df_1["treatment_length"] =
treatment_length = df_1.loc[["Patient_ID"] == positive.loc["Patient_ID"]]
print(df_1)

for n in df:
    if df[["Patient_ID"]].iloc[n] == patients.iloc[n]:
        print(df[["Patient_ID"]])

import matplotlib.pyplot as plt

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

df_1.columns

df_1.loc[50]

df1_index = pd.DataFrame(df_1,
                         index=[["Patient_ID"]],
                         columns=(["Patient_ID", "Drug_admin_date"]))
df1_index
df1_index_copy = df1_index.copy()

df1_index

df_1.columns

type(df_1)

type(df1_index)

import matplotlib.pyplot as plt

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

for i in df1_index.items():
    if i == df1_index:
        print(i)

row = next(df.iterrows())[1]
row
df1_index.keys

index = pd.Index(df1_index, name="Patient_Info")

# deef =  pd.DataFrame(index,
#  index=index,
#   columns=(["Patient_ID", "Drug_admin_date"]))
# deef
index

idd_ = df_1.loc['Patient_ID'] == df['Patient_ID']

print(len(df))
df.columns

# check the shape of the variable
# df_index.shape


# creating an index of the data frame that we can use to sort through the data using Pandas Dataframe function
df_index = pd.DataFrame(df,
                        index=range(0, 37),
                        columns=(["Patient_ID", "drug_390_admin_flag",
                                  "ER_positive", "PR_positive", "HER2_positive"]))
df_index
# it needs to search row by row, at the coordinates to find the value and use that to search the coordinates of
# next columns


# df_index.query( "drug_390_admin_flag"and"PR_positive")
# df_index[( df_index["ER_positive"]< df_index["PR_positive"])]
# df_index["drug_390_admin_flag"]
# df_index[df_index["Patient_ID"].isin([df_index["PR_positive"]])]

df[["PR_positive"]], df[["Patient_ID"]]

patients = df.groupby('PR_positive')
# print(df.groupby(['Patient_ID']).groups)
patients.head()

import matplotlib.pyplot as plt

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# grab the index of the column
# test example
df_index.iloc[36]

df_index.iloc[0:], df.columns.get_indexer(['Patient_ID',
                                           "PR_positive", "ER_positive"])

df_index.iloc[0:], df.columns.get_indexer(['Patient_ID'])

df.loc[[1], ["Patient_ID"]]

positive_tests.columns

tests = positive_tests[positive_tests["Patient_ID"].isin(df_1["Patient_ID"])]

# positive_tests["Patient_ID"].isin(df_1["Patient_ID"])
pos = positive_tests["Patient_ID"].items
pos

df_1["Positives"] = positive_tests["Patient_ID"]
for n in positive_tests["Patient_ID"]:
    print('These patients were positive: ',n)


df_1

# df_1[["Positives"]]= df_1.loc[df_1["Patient_ID"] & positive_tests["Patient_ID"],
#  df_1["Positive"]] = 0
posit = positive_tests["Patient_ID"]
# select items in patient ID
df_1.groupby([["Patient_ID"], posit])

# df_1[df_1['Patient_ID'] == positive_tests[['Patient_ID']].items]
df_1['Patient_ID']

df.iloc[[22], df.columns.get_indexer(['Patient_ID',
                                      "PR_positive", "ER_positive"])], df.iloc[
    [0], df.columns.get_indexer(['Patient_ID',
                                 "PR_positive", "ER_positive"])]

df.values

# criterion = df['Patient_ID'].map(lambda x: x.startswith(1))
positive_tests = {'drug_390_admin_flag': [1], "PR_positive": [1], "ER_positive": [1], "HER2_positive": [1]}
# s= df.isin(positive_tests).any(1)
df.query('a < b & b < c')

# data frame with data from the first sheet
# df = pd.read_excel('/Users/michaelnaylor/Downloads/PatientData_ProgrammingAssignment.xlsx',
#  true_values = None, keep_default_na = True, na_filter=False,
#  verbose=True, header = 1
#  ,names=None, index_col=None,
#  sheet_name = 0, #sheet no in excel doc,
#  skiprows=2)


# df.head() #print the first 5 values to see if data has loaded properly
# df.dropna() #drop the NaN values in the dataset
# df.dropna(axis='columns')
# df.head() #shows the first 5 elements
#

# drug_data = df.values
# drug_data

df_Z = pd.DataFrame(
    index=list('012345'))  # ,
# columns=list('Patient_ID','ER_positive'))
df_Z

# drug_data[[
type(df)

# df[["Patient_ID"]]
patient_ID = df[["Patient_ID"]].values
# print(df.columns) #show the column names
# df.columns[2] #show the label of column in index 2
print(len(patient_ID))
print(patient_ID[36], patient_ID[4])

# df[["Patient_ID"]].head()
hist = df[["drug_390_admin_flag"]].hist(grid=True, bins=20)  # creating a histogram to visualise the data
hist = df[["ER_positive"]].hist(grid=True, bins=20)
df[["ER_positive"]]

for n in df[["ER_positive"]]:
    if n == 1:
        print(n)

drug_admin = df.loc[df["drug_390_admin_flag"] >= 1]
len(drug_admin)

er_Positive = df.loc[df["ER_positive"] >= 1]  # print rows with column values 1 or greater
len(er_Positive)

pr_Positive = df.loc[df["PR_positive"] >= 1]
len(pr_Positive)

df.count()

df_1["Patient_ID"].count()

pr_Positive.to_numpy()

pr_Positive["Patient_ID"].isin(df_1["Patient_ID"])  # check if the values in pr_positive are in df_1

positive_IDs = pr_Positive.groupby('Patient_ID')
for n, name in positive_IDs:
    print(n)
# if n == df_1[["Patient_ID"]]:
#  print(df_1["drug_admin_date"])

positive_IDs.dtype

positive_tests

er_Negative = df.loc[df["ER_positive"] == 0]
len(er_Negative)

pr_Negative = df.loc[df["PR_positive"] == 0]
len(pr_Negative)

positive_tests = [pr_Positive, er_Positive]  # creates a list
len(positive_tests)

positive_tests.i

positive_tests = df.loc[(df["PR_positive"] == 1) | (df["ER_positive"] == 1)]

positive_tests

patient_data_copy.loc[, Drug_admin_date]- patient_data_copy.loc[, :].shift(1)

patient_data_copy.xs(level='Patient_ID')

patient_data_copy.iloc[1:, ]

pizza = [n for n in positive_tests["Patient_ID"] for y in df_1["Patient_ID"] if n == y]
print(len(pizza), pizza)

df_1copy = df_1.copy()
df_1copy

df_1["Positives"] = positive_tests["Patient_ID"]
for n in positive_tests["Patient_ID"]

df_1copy["length"] =  # [df_1['Patient_ID'].duplicated() == True for n in positive_tests["Patient_ID"]
for y in df_1["Patient_ID"] if n == y]= df_1[["Drug_admin_date"]] - df_1[["Drug_admin_date"]].shift(1)

df_1copy["length"] =['Positive'
for n in positive_tests["Patient_ID"]
    for y in df_1["Patient_ID"]
        if n == y]  # df_1[["Drug_admin_date"]] - df_1[["Drug_admin_date"]].shift(1)

        df_1["Patient_ID"].value_counts()

        len(df_1["Patient_ID"])

        df_1.loc[df_1["Patient_ID"] == 9489]

        df_1["Positive"] =[n for n in positive_tests["Patient_ID"] for y in df_1["Patient_ID"] if n == y]

        positive_tests.columns

        pr_Negative["Patient_ID"].isin(df_1["Patient_ID"])

        negative_tests = df.loc[(df["PR_positive"] == 0) & (df["ER_positive"] == 0)]

        # positive_tests.values
        df['Patient_ID'].values

        len(df)-len(positive_tests)

        # loop through the patients with positive tests
        for n in positive_tests["Patient_ID"]:
        # for ide in df_1[['Patient_ID']]:
        # while df[n] == df_1[ide]:
            print("Patient_ID:", n)
    print("There are:", len(positive_tests), "positive tests")

    # create class to sort patient information


class Patient_Info:
    def __init__(self, patient_ID, diagnosis, treatment_length):
        self.patient_ID = patient_ID
        self.diagnosis = diagnosis
        self.treatment_length = treatment_length


negative_tests

grp1 = negative_tests.groupby('Patient_ID')
for n, name in grp1:
    print(n)
# print(name)

# print()

positive_tests_0 = df.loc[df["Patient_ID"].isin(positive_tests)]
positive_tests_0

df.loc[:"Patient_ID"]

# creating a function to loop through the values of df["Patient_ID"]
# def treatment_length():
for n in df["Patient_ID"].values:
    for num in range(len(positive_tests)):
        if positive_tests[num] == positive_tests:
            print(positive_tests[num])
            Q

positive_tests

positive_tests[0]

df.loc[["Patient_ID"]]

total_positives = len(positive_tests[0]) + len(positive_tests[1])
total_positives

# need to create a class that allows you to store the patient ID values as the value to be called.
# The function should give you the ability to search for an ID by imputting a list of values
# further consideration should be kept for interaction, in the UI such as a searchbar that attempts to predict
# the number that will be entered. The function can only accept INTs that are 4 numbers long.
type(df[['Patient_ID']])

df_1 = pd.read_excel('/Users/michaelnaylor/Downloads/PatientData_ProgrammingAssignment.xlsx',
                     true_values=None, keep_default_na=True, na_filter=False,
                     verbose=True, header=1
                     , names=None, index_col=None,
                     sheet_name=1, skiprows=2)
df_1.head()

df_1[['Drug_admin_date']], df_1[['Patient_ID']]

df_1copy

features = ["Patient_ID", "Drug_admin_date"]
print(df_1[features])

# for ID in features:

# function to calculate the length of a patients treatment

# def treatmenttime():


df['drug_390_admin_flag'].values
print("Column Name:", df.columns[2], "\nValues:", df['drug_390_admin_flag'].values,
      "\nNo. of Values:", len(df['drug_390_admin_flag'])
      , "\nMean:", sum(df["drug_390_admin_flag"]) / len(df["drug_390_admin_flag"]))

df

data = df.iloc[:, 2].values
data

print("First 5 numbers in this ndarray:", data[0:5])  # data[1])

# we need a function that compares the key to the value of the vectors in each column
patient_ID = df_1['Patient_ID'].values
drug_admin_date = df_1['Drug_admin_date'].values
patient_ID

# df_1[Patient_ID]
# create a function which counts how many times an ID appears in the array,
# then arranges the  patient ID in sequential order, smallest to largest without repetitions
patient_ID[0]

# for x, ID in np.ndenumerate(patient_ID):
# print(x, ID)


# test out the patient ID array
patient_ID[0]

# test out drug admin date data
drug_admin_date[0]

# variable to store the treatment length
treatment_length = np.datetime64(drug_admin_date[0], 'W') - np.datetime64(drug_admin_date[1], 'W')
print("Patient", Patient_ID[0], "had treatment for", treatment_length, "Weeks")

print("This is a", type(data), "\nShowing a 1D array:", data, "\nThere are", len(data), "examples in this dataset")

data = df.iloc[:, 1].values

print(type(data), data)

data = df.iloc[:, 3].values

print(type(data), data)

data = df.iloc[:, 4].values
print(type(data), data)

# print(df.describe)

df.dropna()

print(df.describe)

print(df['Patient_ID'], df['drug_390_admin_flag'], df['ER_positive'])
# df

old = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0])

old_mean = (sum(old) / len(old))  # get the average of the old data

print(old_mean)



