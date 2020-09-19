import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
plt.close('all')
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../data/raw/Shakespeare_data.csv')

df = df.dropna()
df = df[['Play', 'PlayerLinenumber', 'ActSceneLine', 'Player', 'PlayerLine']]

actSceneLine = df['ActSceneLine'].str.split('.', n = 2, expand = True)
df['Act'] = pd.to_numeric(actSceneLine[0])
df['Scene'] = pd.to_numeric(actSceneLine[1])
df['Line'] = pd.to_numeric(actSceneLine[2])
df = df[['Play', 'PlayerLinenumber', 'Act', 'Scene', 'Line', 'Player', 'PlayerLine']]

df = pd.get_dummies(df, columns=['Play'])

df['Player'] = df['Player'].astype('category').cat.codes

print(df.describe())
print(df.corr())

macbeth = df[df['Play_macbeth'] == 1]
macbeth.plot.scatter(x='PlayerLinenumber', y='Player', title='Player vs Player Line Number')
plt.show()

df.pop('Line')

dfWordless = df.copy()

df['PlayerLine'] = df['PlayerLine'].str.replace('.', '')
df['PlayerLine'] = df['PlayerLine'].str.replace('!', '')
df['PlayerLine'] = df['PlayerLine'].str.replace('?', '')
df['PlayerLine'] = df['PlayerLine'].str.replace(',', '')
df['PlayerLine'] = df['PlayerLine'].str.replace(':', '')
df['PlayerLine'] = df['PlayerLine'].str.replace(';', '')
df['PlayerLine'] = df['PlayerLine'].str.replace('-', ' ')
df['PlayerLine'] = df['PlayerLine'].str.lower()

frequentWords = df['PlayerLine'].str.split(expand=True).stack().value_counts().head(100)

df['PlayerLine'] = ' ' + df['PlayerLine'] + ' '
for word, count in frequentWords.items():
    df[word] = df['PlayerLine'].str.count(' ' + word + ' ')

df.to_csv('../data/processed/Shakespeare_Transformed_WithWordAnalysis.csv')
dfWordless.to_csv('../data/processed/Shakespeare_Transformed_Wordless.csv')

playerCol = df.pop('Player')
df.insert(0, 'Player', playerCol)

dfWordless.pop('Player')
dfWordless.insert(0, 'Player', playerCol)

df.pop('PlayerLine')
array = df.values
X = array[:, 1:]
Y = array[:, 0]

dfWordless.pop('PlayerLine')
arrayWordless = dfWordless.values
XWordless = arrayWordless[:, 1:]
YWordless = arrayWordless[:, 0]

X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=0.3, shuffle=True)
X_trainWordless, X_validateWordless, Y_trainWordless, Y_validateWordless = train_test_split(XWordless, YWordless, test_size=0.3, shuffle=True)

kFold = StratifiedKFold(n_splits=10, shuffle=True)

print('Decision Tree With Word Analysis')
decisionTreePredictions = cross_val_predict(DecisionTreeClassifier(), X_train, Y_train, cv=kFold)
print(accuracy_score(Y_train, decisionTreePredictions))

print('Random Forest With Word Analysis')
randomForestPredictions = cross_val_predict(RandomForestClassifier(n_estimators=20), X_train, Y_train, cv=kFold)
print(accuracy_score(Y_train, randomForestPredictions))

print('K Neighbors With Word Analysis')
kNeighborsPredictions = cross_val_predict(KNeighborsClassifier(), X_train, Y_train, cv=kFold)
print(accuracy_score(Y_train, kNeighborsPredictions))

print('Decision Tree Without Word Analysis')
decisionTreeWordlessPredictions = cross_val_predict(DecisionTreeClassifier(), X_trainWordless, Y_trainWordless, cv=kFold)
print(accuracy_score(Y_trainWordless, decisionTreeWordlessPredictions))

print('Random Forest Without Word Analysis')
randomForestWordlessPredictions = cross_val_predict(RandomForestClassifier(n_estimators=30), X_trainWordless, Y_trainWordless, cv=kFold)
print(accuracy_score(Y_trainWordless, randomForestWordlessPredictions))

print('K Neighbors Without Word Analysis')
kNeighborsWordlessPredictions = cross_val_predict(KNeighborsClassifier(), X_trainWordless, Y_trainWordless, cv=kFold)
print(accuracy_score(Y_trainWordless, kNeighborsWordlessPredictions))

model = DecisionTreeClassifier()
model.fit(X_trainWordless, Y_trainWordless)

predictions = model.predict(X_validateWordless)

print('Validation of Decision Tree Without Word Analysis')
print(accuracy_score(Y_validateWordless, predictions))

print(classification_report(Y_validateWordless, predictions))