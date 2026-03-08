#Set-up
-Folder created - titanic_assignment
-Structure defined for data (csv files), notebooks (for Jupyter) and scripts. 
-Downloaded and saved train.csv and test.csv at data subfolder

Code: 

import pandas as pd
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

train_df.info()
train_df.head() #testing the code

# Data Cleaning

##Missing value handling
###Method used
train_df.isnull().sum()
    -is.null() returns true for every missing variable
    -sum() counts and outputs the number of true for each column
###Solutions chosen
Age - since age is normally distributed, imputing using median would fill the empty columns
	-Code
	train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
	train_df['Age'].head(10)

2. Cabin - due to missing values, extrapolate first letter of cabin for all values, drop the numbers, 
then impute using mode or median to fill the missing ones. Cabin is therefore dropped, and a new column called Deck is used.
	-Code
	train_df['Deck'] = train_df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')
	train_df['Deck'].value_counts()
	
	train_df = train_df.drop('Cabin', axis=1)
	train_df.isnull().sum()
3. Embarked - since only two are missing, impute using mode (assumption is made that the busiest port made the missing entry errors, minor effect to dataset is expected)
	-Code
	train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
	train_df['Embarked'].value_counts()
