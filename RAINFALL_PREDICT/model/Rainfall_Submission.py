import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("data/train.csv")

i,j = 0, 0
fig, ax = plt.subplots(4, 3, figsize=(18 , 11))
for colum in data.columns[:-1]:
    sns.histplot(data=data, x=colum, hue="rainfall", kde=True, ax=ax[j, i])
    if i < 2:
        i += 1
    else:
        i = 0
        j += 1
    plt.subplots_adjust(hspace = 0.5)

data.shape

plt.figure(figsize=(12, 10))
sns.heatmap(data=data.corr(), annot=True)

#data.head(8)
#data.describe()
#data.info()
#data.shape

data['season'] = data['day'] % 365

def get_season(day):
    month = (day % 365) // 30 + 1
    if month in [12, 1, 2]:
        return 0 #'Winter'
    elif month in [3, 4, 5]:
        return 1 #'Spring'
    elif month in [6, 7, 8]:
        return 2 #'Summer'
    else:
        return 3 #'Autumn'

data['season'] = data['day'].apply(get_season)
data = pd.get_dummies(data, columns=['season'], dtype=int)
data['temp_range'] = data['maxtemp'] - data['mintemp']
data['temp_dew_diff'] = data['temparature'] - data['dewpoint']

data, y = data.drop(["rainfall", "day", "id", "mintemp", "maxtemp"], axis=1), data["rainfall"]

norm = StandardScaler()
data2 = norm.fit_transform(data)
data = pd.DataFrame(data=data2, columns=data.columns)

RFS = RandomForestClassifier(n_estimators=25, min_samples_leaf=1, min_samples_split=50, max_depth=5)
RFS.fit(data, y)

test_data = pd.read_csv("data/test.csv")
test_data['season'] = test_data['day'] % 365

def get_season(day):
    month = (day % 365) // 30 + 1
    if month in [12, 1, 2]:
        return 0 #'Winter'
    elif month in [3, 4, 5]:
        return 1 #'Spring'
    elif month in [6, 7, 8]:
        return 2 #'Summer'
    else:
        return 3 #'Autumn'

test_data['season'] = test_data['day'].apply(get_season)
test_data = pd.get_dummies(test_data, columns=['season'], dtype=int)
test_data['temp_range'] = test_data['maxtemp'] - test_data['mintemp']
test_data['temp_dew_diff'] = test_data['temparature'] - test_data['dewpoint']
test_data = test_data.drop(["id", "day", "mintemp", "maxtemp"], axis=1)
test_data = pd.DataFrame(data=norm.transform(test_data), columns=test_data.columns)

#test_data.info()
#data.info()
#test_data.head(5)

test_data.fillna(0, inplace=True)

sub = pd.read_csv("data/sample_submission.csv")
sub["rainfall"] = 1 - RFS.predict_proba(test_data)
sub.to_csv("answer.csv", index=False)