\#Set-up

-Folder created - titanic\_assignment

-Structure defined for data (csv files), notebooks (for Jupyter) and scripts. 

-Downloaded and saved train.csv and test.csv at data subfolder



Code: 



import pandas as pd

train\_df = pd.read\_csv('../data/train.csv')

test\_df = pd.read\_csv('../data/test.csv')



train\_df.info()

train\_df.head() #testing the code



\#Step 1: Data Cleaning



\##Missing value handling

\###Method used

train\_df.isnull().sum()





