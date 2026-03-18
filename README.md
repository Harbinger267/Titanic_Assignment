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


\#Section 2: Feature Engineering

Feature engineering - creating new columns from the cleaned dataset so the model can capture patterns that are not obvious from the raw values alone.

\## 2.1: Family based features

\###Method used

Passengers travelling with relatives may behave differently from passengers travelling alone.

New columns:

1. FamilySize = SibSp + Parch + 1
2. IsAlone = 1 if FamilySize == 1, otherwise 0

\###Code

feature\_df = pd.read\_csv('../data/train\_cleaned.csv')

feature\_df\['FamilySize'] = feature\_df\['SibSp'] + feature\_df\['Parch'] + 1

feature\_df\['IsAlone'] = (feature\_df\['FamilySize'] == 1).astype(int)

feature\_df\[\['SibSp','Parch','FamilySize','IsAlone']].head()


\## 2.2: Title grouping

\###Method used

Titles already extracted in Part 1 can be simplified further so uncommon spellings map to common social groups.

\###Code

feature\_df\['Title'] = feature\_df\['Title'].replace({

    'Mlle': 'Miss',

    'Ms': 'Miss',

    'Mme': 'Mrs'

})

feature\_df\['Title'].value\_counts()

Reason - reduces unnecessary category fragmentation and keeps title information useful for modelling.


\## 2.3: Age grouping

\###Method used

Instead of only using exact ages, group ages into broader life stages.

Groups used:

1. Child = 0 - 12
2. Teen = 13 - 19
3. Adult = 20 - 59
4. Senior = 60+

\###Code

feature\_df\['AgeGroup'] = pd.cut(

    feature\_df\['Age'],

    bins=\[0, 12, 19, 59, float('inf')],

    labels=\['Child','Teen','Adult','Senior'],

    include\_lowest=True

)

feature\_df\[\['Age','AgeGroup']].sample(10)

Reason - grouped ages may capture survival patterns more clearly than raw continuous age values.


\## 2.4: Ticket prefix and fare-per-person

\###Method used

Ticket values contain prefixes that may reflect booking class or ticket type. Fare is also adjusted by family size to estimate how much fare corresponds to each passenger.

\###Code

feature\_df\['TicketPrefix'] = (

    feature\_df\['Ticket']

    .astype(str)

    .str.replace(r'\\d+', '', regex=True)

    .str.replace(r'[./]', '', regex=True)

    .str.replace(' ', '', regex=False)

    .replace('', 'NUM')

)

feature\_df\['FarePerPerson'] = feature\_df\['Fare'] / feature\_df\['FamilySize']

feature\_df\[\['Ticket','TicketPrefix','Fare','FamilySize','FarePerPerson']].head()

Reason - these variables can give more context than Ticket and Fare alone.


\## 2.Final: Storing engineered dataset in a separate csv called train\_featured.csv

feature\_df.to\_csv('../data/train\_featured.csv', index=False)

featured\_df = pd.read\_csv('../data/train\_featured.csv')

featured\_df.head()


\#Section 3: Feature Selection

Feature selection - choosing the most useful columns for prediction while removing unnecessary or overly noisy variables.

\## 3.1: Choosing candidate columns

Original text columns such as Name and Ticket are not directly used for modelling. Instead, use cleaned and engineered columns that summarise them better.

Candidate columns:

1. Pclass
2. Sex
3. Age
4. Fare\_log
5. Embarked
6. Deck
7. Title
8. FamilySize
9. IsAlone
10. AgeGroup
11. FarePerPerson


\## 3.2: Encoding categorical values

\###Method used

Machine learning models usually require numeric inputs, so categorical columns are one-hot encoded.

\###Code

model\_df = featured\_df\[\[

    'Survived','Pclass','Sex','Age','Fare\_log','Embarked','Deck',

    'Title','FamilySize','IsAlone','AgeGroup','FarePerPerson'

\]\].copy()

model\_df = pd.get\_dummies(

    model\_df,

    columns=\['Sex','Embarked','Deck','Title','AgeGroup'],

    drop\_first=True

)

model\_df.head()


\## 3.3: Ranking features

\###Method used

Use correlation and mutual information to identify variables that carry the most predictive signal for survival.

\###Code

X = model\_df.drop('Survived', axis=1)

y = model\_df\['Survived']

correlations = model\_df.corr(numeric\_only=True)\['Survived'].sort\_values(ascending=False)

correlations

from sklearn.feature\_selection import SelectKBest, mutual\_info\_classif

selector = SelectKBest(score\_func=mutual\_info\_classif, k=10)

selector.fit(X, y)

selected\_features = X.columns\[selector.get\_support()]

selected\_features

Expected strong features from this dataset include Sex, Pclass, Fare\_log, Title, Deck, and IsAlone / FamilySize related columns.


\## 3.4: Final selected dataset

\###Code

selected\_df = pd.concat(\[y, X\[selected\_features]], axis=1)

selected\_df.head()

selected\_df.to\_csv('../data/train\_selected.csv', index=False)


\## 3.Final: Summary

1. Feature engineering created FamilySize, IsAlone, AgeGroup, TicketPrefix and FarePerPerson.
2. Title values were standardised further for cleaner categories.
3. Feature selection reduced the dataset to the most informative model-ready variables.
4. Final output is saved as train\_selected.csv for use in later modelling tasks.

