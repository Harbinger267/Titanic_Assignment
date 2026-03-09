\#Set-up

-Folder created - titanic\_assignment

-Structure defined for data (csv files), notebooks (for Jupyter) and scripts.

-Downloaded and saved train.csv and test.csv at data subfolder



Code:



import pandas as pd

train\_df = pd.read\_csv('../data/train.csv')

test\_df = pd.read\_csv('../data/test.csv')



train\_df.head() #testing the code



\#Section 1: Data Cleaning



\## 1.1: Missing value handling

\###Method used

train\_df.isnull().sum()

    -is.null() returns true for every missing variable

    -sum() counts and outputs the number of true for each column

\###Solutions chosen

1. Age - since age is normally distributed, imputing using median would fill the empty columns

 	-Code

 	train\_df\['Age'] = train\_df\['Age'].fillna(train\_df\['Age'].median())

 	train\_df\['Age'].head(10)



2\. Cabin - due to missing values, extrapolate first letter of cabin for all values, drop the numbers,

then impute using mode or median to fill the missing ones. Cabin is therefore dropped, and a new column called Deck is used.

 	-Code

 	train\_df\['Deck'] = train\_df\['Cabin'].apply(lambda x: str(x)\[0] if pd.notnull(x) else 'U')

 	train\_df\['Deck'].value\_counts()

 

 	train\_df = train\_df.drop('Cabin', axis=1)

 	train\_df.isnull().sum()

3\. Embarked - since only two are missing, impute using mode (assumption is made that the busiest port made the missing entry errors, minor effect to dataset is expected)

 	-Code

train\_df\['Embarked'] = train\_df\['Embarked'].fillna(train\_df\['Embarked'].mode()\[0])train\_df\['Embarked'].value\_counts()



\## 1.2: Outlier Handling

Outlier - value in the dataset that is significantly different from the rest, skews regular distribution of the dataset. Usually only numerical values are affected by outliers.

\###Finding outliers:

Interquartile Range method - Standard method that uses quartiles to find outliers in the dataset. 

We use a looping method for only numerical values in the code to find the number of outliers and output in a readable format.



&nbsp;	numeric\_cols = train\_df.select\_dtypes(include=\['int64','float64']).columns

&nbsp;	for col in numeric\_cols:

&nbsp;   		Q1 = train\_df\[col].quantile(0.25)

&nbsp;   		Q3 = train\_df\[col].quantile(0.75)

&nbsp;   		IQR = Q3 - Q1

&nbsp;   

&nbsp;   		outliers = train\_df\[(train\_df\[col] < Q1 - 1.5\*IQR) | (train\_df\[col] > Q3 + 1.5\*IQR)]

&nbsp;   

&nbsp;   		print(col, "outliers:", len(outliers))

From the code above, outliers are found in Age, SibSp, Parch and Fare.

Age, SibSp (siblings or spouses) and Parch(parents and children) are all legitimate values, no matter how irregular, therefore are ignored for outlier handling. (will later be used in feature engineering)

\#### Fare handling - improve stability of models that handle distance-based data or handle linear regression

1. Capping at 99th percentile - Fare above the value at 99th percentile is replaced with that value, improves model stability 
2. Log transform - Creating a new value Fare\_log = log(1+Fare) slightly changing the scale and reducing the skew of the dataset.

&nbsp;	-Code

\# Cap extreme fares

upper\_limit = train\_df\['Fare'].quantile(0.99)

train\_df\['Fare'] = np.where(train\_df\['Fare'] > upper\_limit, upper\_limit, train\_df\['Fare'])



\# Reduce skew

train\_df\['Fare\_log'] = np.log1p(train\_df\['Fare'])



Results - using train\_df\['Fare'].describe() or train\_df.info()

&nbsp;	- creating a histogram of the data to show distribution visually

&nbsp;	import matplotlib.pyplot as plt



&nbsp;	plt.figure(figsize=(10,4))



&nbsp;	plt.subplot(1,2,1)

&nbsp;	train\_df\['Fare'].hist(bins=50)

&nbsp;	plt.title('Capped Fare')



&nbsp;	plt.subplot(1,2,2)

&nbsp;	train\_df\['Fare\_log'].hist(bins=50)

&nbsp;	plt.title('Log-transformed Fare')



&nbsp;	plt.show()



\## 1.3: Handling inconsistencies and duplicates
Duplicates - repeated rows or values

Finding duplicates:

train\_df.duplicated().sum()

Output is np.int64(0) => NO duplicate rows in the dataset



Inconsistencies - can happen due to case mismatching e.g Male, male, MALE, C or c.

Usually in string based data: sex, embarked and name.

Finding :

train\_df\['Sex'].value\_counts()

-Output shows all values are either male or female => NO inconsistencies

train\_df\['Embarked'].value\_counts()

-Output shows all values are either S, C or Q => NO inconsistencies

For name, only interested in finding unique values such as the title e.g Mr, Miss, Mrs etc.

Sample data to see inconsistencies:

train\_df\['Name'].head(10)        

train\_df\['Name'].sample(10)   

Extracting the titles:

train\_df\['Title'] = train\_df\['Name'].str.extract(' (\[A-Za-z]+)\\.', expand=False)

train\_df\['Title'].value\_counts()

Replacing rare titles such as Mme, Jonkheer

rare\_titles = \['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']

train\_df\['Title'] = train\_df\['Title'].replace(rare\_titles, 'Rare')



\## 1.Final: Storing cleaned dataset in a separate csv called train\_cleaned.csv

train\_df.to\_csv('../data/train\_cleaned.csv', index=False)

Loading and viewing to confirm it works

cleaned\_df = pd.read\_csv('../data/train\_cleaned.csv')

cleaned\_df.head()

