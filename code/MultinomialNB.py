import random
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from scipy import sparse
from warnings import filterwarnings

filterwarnings('ignore')

from sklearn import model_selection
from sklearn.preprocessing import  MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import log_loss, classification_report, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from sklearn.naive_bayes import MultinomialNB


train_df = pd.read_json('../store/train.json.zip', compression='zip')
test_df = pd.read_json('../store/train.json.zip', compression='zip')

print(f"<DataLoding> train_set: {train_df.shape}")
print(f"<DataLoding> test_set: {test_df.shape}")

# Reduce Outliers
print(f"<DataPreprocessing> Start Reduce Outliers...")

mean_price = int(train_df['price'].mean())
test_df.loc[test_df['price'] < 200, 'price'] = mean_price
train_df.loc[train_df['price'] < 200, 'price'] = mean_price

# Data preprocessing
print(f"<DataPreprocessing> Start Merge Synonyms...")
train_test = pd.concat([train_df, test_df], 0, sort=False)

features = train_test[["features"]].apply(
    lambda _: [list(map(str.strip, map(str.lower, x))) for x in _])

n = 5
feature_counts = Counter()
for feature in features.features:
    feature_counts.update(feature)
feature = sorted([k for (k, v) in feature_counts.items() if v > n])
feature[:10]


def clean(s):
    x = s.replace("-", "")
    x = x.replace(" ", "")
    x = x.replace("24/7", "24")
    x = x.replace("24hr", "24")
    x = x.replace("24-hour", "24")
    x = x.replace("24hour", "24")
    x = x.replace("24 hour", "24")
    x = x.replace("common", "cm")
    x = x.replace("concierge", "doorman")
    x = x.replace("bicycle", "bike")
    x = x.replace("pets:cats", "cats")
    x = x.replace("allpetsok", "pets")
    x = x.replace("dogs", "pets")
    x = x.replace("private", "pv")
    x = x.replace("deco", "dc")
    x = x.replace("decorative", "dc")
    x = x.replace("onsite", "os")
    x = x.replace("outdoor", "od")
    x = x.replace("ss appliances", "stainless")
    return x


def feature_hash(x):
    cleaned = clean(x, uniq)
    key = cleaned[:4].strip()
    return key

def plot_learning_curve(estimator, title, X, y,
                        ylim=None, cv=None, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                            cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt

key2original = defaultdict(list)
k = 4
for f in feature:
    cleaned = clean(f)
    key = cleaned[:k].strip()

    key2original[key].append(f)


def to_tuples():
    for f in feature:
        key = clean(f)[:k].strip()
        yield (f, key2original[key][0])


deduped = list(to_tuples())
df = pd.DataFrame(deduped, columns=["original_feature", "unique_feature"])
dict_rep_features = pd.Series(df['unique_feature'].values, df['original_feature'].values)

# Data preprocessing
print(f"<DataPreprocessing> Optimize the Features Field..")
test_df['features'] = test_df['features'].apply(lambda x: list(map(str.strip, map(str.lower, x)))) \
    .apply(lambda x: [dict_rep_features[i] for i in x if i in dict_rep_features.index]) \
    .apply(lambda x: list(set(x)))

train_df['features'] = train_df['features'].apply(lambda x: list(map(str.strip, map(str.lower, x)))) \
    .apply(lambda x: [dict_rep_features[i] for i in x if i in dict_rep_features.index]) \
    .apply(lambda x: list(set(x)))

features_to_use = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

# Target Encoding
print(f"<Feature Engineering> Start Target Encoding...")
start_values = [0, 0, 0]

index = list(range(train_df.shape[0]))
random.shuffle(index)
a = [np.nan] * len(train_df)
b = [np.nan] * len(train_df)
c = [np.nan] * len(train_df)

for i in range(5):
    building_level = {}
    for j in train_df['manager_id'].values:
        building_level[j] = start_values.copy()
    test_index = index[int((i * train_df.shape[0]) / 5):int(((i + 1) * train_df.shape[0]) / 5)]
    train_index = list(set(index).difference(test_index))
    for j in train_index:
        temp = train_df.iloc[j]
        if temp['interest_level'] == 'low':
            building_level[temp['manager_id']][0] += 1
        if temp['interest_level'] == 'medium':
            building_level[temp['manager_id']][1] += 1
        if temp['interest_level'] == 'high':
            building_level[temp['manager_id']][2] += 1
    for j in test_index:
        temp = train_df.iloc[j]
        if sum(building_level[temp['manager_id']]) != 0:
            a[j] = building_level[temp['manager_id']][0] * 1.0 / sum(building_level[temp['manager_id']])
            b[j] = building_level[temp['manager_id']][1] * 1.0 / sum(building_level[temp['manager_id']])
            c[j] = building_level[temp['manager_id']][2] * 1.0 / sum(building_level[temp['manager_id']])
train_df['manager_level_low'] = a
train_df['manager_level_medium'] = b
train_df['manager_level_high'] = c

a = []
b = []
c = []
building_level = {}
for j in train_df['manager_id'].values:
    building_level[j] = start_values.copy()
for j in range(train_df.shape[0]):
    temp = train_df.iloc[j]
    if temp['interest_level'] == 'low':
        building_level[temp['manager_id']][0] += 1
    if temp['interest_level'] == 'medium':
        building_level[temp['manager_id']][1] += 1
    if temp['interest_level'] == 'high':
        building_level[temp['manager_id']][2] += 1

for i in test_df['manager_id'].values:
    if i not in building_level.keys():
        a.append(np.NaN)
        b.append(np.NaN)
        c.append(np.NaN)
    else:
        a.append(building_level[i][0] * 1.0 / sum(building_level[i]))
        b.append(building_level[i][1] * 1.0 / sum(building_level[i]))
        c.append(building_level[i][2] * 1.0 / sum(building_level[i]))
test_df['manager_level_low'] = a
test_df['manager_level_medium'] = b
test_df['manager_level_high'] = c

features_to_use.append('manager_level_low')
features_to_use.append('manager_level_medium')
features_to_use.append('manager_level_high')

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# Count Encoding
print(f"<Feature Engineering> Start Count Encoding...")
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

train_df["created"] = pd.to_datetime(train_df["created"])
test_df["created"] = pd.to_datetime(test_df["created"])

train_df["created_year"] = train_df["created"].dt.year
test_df["created_year"] = test_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
test_df["created_month"] = test_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
test_df["created_day"] = test_df["created"].dt.day
train_df["created_hour"] = train_df["created"].dt.hour
test_df["created_hour"] = test_df["created"].dt.hour

# Label Encoding
print(f"<Feature Engineering> Start Label Encoding...")
categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
    if train_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values) + list(test_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values))
        test_df[f] = lbl.transform(list(test_df[f].values))
        features_to_use.append(f)

features_to_use.extend(
    ["num_photos", "num_features", "num_description_words", "created_year", "created_month", "created_day",
     "listing_id", "created_hour"])





print(f"<Feature Engineering> Inverse Document Frequency...")
train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])

train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high': 2, 'medium': 1, 'low': 0}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

print(f"<Feature Engineering> Finished Feature Engineering: train_set{train_df.shape}")
print(f"<Feature Engineering> Finished Feature Engineering: test_set{test_df.shape}")

print(f"<Training Module> Select Module...")


scaler = MinMaxScaler()
train_X = scaler.fit_transform(train_df[features_to_use])
test_X = scaler.transform(test_df[features_to_use])
fea_x = train_X
fea_y = train_y


print(f"<Training Module> Start Cross-Validation...")
test_pred = None
log_loss_score = []

title = "Learning Curves (Naive Bayes)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = MultinomialNB(alpha=0.1)
scores = model_selection.cross_val_score(estimator, train_X, train_y, cv=10)
# print(scores)
y_pred = model_selection.cross_val_predict(estimator, train_X, train_y, cv=10)
target_names = ['manager_level_low', 'manager_level_medium', 'manager_level_high']
print(classification_report(train_y, y_pred, target_names=target_names))

plot_learning_curve(estimator, title, fea_x, fea_y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.show()

clf = MultinomialNB(alpha=0.1)
finalScore = [[0 for col in range(6)] for row in range(10)]
n = 1
for train_idx, test_idx in model_selection.KFold(n_splits=10, shuffle=True, random_state=2020).split(train_X, train_df[
    'interest_level']):

    clf.fit(train_X[train_idx], train_y[train_idx])
    precision_score_cal = precision_score(train_y[test_idx], clf.predict(train_X[test_idx]), average='weighted')
    recall_score_cal = recall_score(train_y[test_idx], clf.predict(train_X[test_idx]), average='weighted')
    f1_score_cal = f1_score(train_y[test_idx], clf.predict(train_X[test_idx]), average='weighted')
    score_for_train = clf.score(train_X[train_idx], train_y[train_idx])
    score_for_test = clf.score(train_X[test_idx], train_y[test_idx])
    # print(f"precision_score for train_idx {n}   ---- {precision_score_cal}")
    # print(f"recall_score for train_idx {n}      ---- {recall_score_cal}")
    # print(f"f1_score for train_idx {n}          ---- {f1_score_cal}")
    # print(f"score for train_idx {n}             ---- {score_for_train}")
    # print(f"score for test_idx {n}              ---- {score_for_test}")
    finalScore[n-1][0] =precision_score_cal
    finalScore[n-1][1] =recall_score_cal
    finalScore[n-1][2] =f1_score_cal
    finalScore[n-1][3] =score_for_train
    finalScore[n-1][4] =score_for_test

    print(f"<Training Module> Training Data...")
    loss = log_loss(train_y[test_idx], clf.predict_proba(train_X[test_idx]))
    log_loss_score.append(loss)
    finalScore[n - 1][5] = loss
    n += 1
    print(f"<Training Module> Calulate Log_loss...: [Logloss] ---- {loss}")
    if test_pred is None:
        test_pred = clf.predict_proba(test_X)
    else:
        test_pred += clf.predict_proba(test_X)

test_pred /= 10

print()
print("finalScore:  ",finalScore)
print(f"<Training Module> [Final Result]>>>>>>[logloss]: {min(log_loss_score)}")
print(f"<Training Module> Output Data...")
out_df = pd.DataFrame(test_pred)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("MultinomialNB.csv", index=False)
