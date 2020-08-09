import os
import sys
import operator
import random
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# from xgboost import XGBClassifier
# from lightgbm import LGBMRegressor

##############################
# read data
##############################

# train_df = pd.read_json('../input/two-sigma-connect-rental-listing-inquiries/train.json.zip', compression='zip')
# test_df = pd.read_json('../input/two-sigma-connect-rental-listing-inquiries/test.json.zip', compression='zip')

## colab
train_df = pd.read_json('train.json')
test_df = pd.read_json('test.json')

print(f"<DataLoding> train_set: {train_df.shape}")
print(f"<DataLoding> test_set: {test_df.shape}")



##############################
#     Data Preprocessing
##############################

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
# feature[:10]


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
    cleaned = clean(x)
    key = cleaned[:4].strip()
    return key


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


##############################
# Feature engineering
##############################

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

# 将features字段根据词频转换成tf_idf特征矩阵
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



##############################
# learning_curve & overfitting
##############################
print(f"<Hyperparameter Analysis> Select the Module and Start the Analysis...")
fea_x = train_df[features_to_use].values
fea_y = train_y

train_fea_x, test_fea_x, train_fea_y, test_fea_y = model_selection.train_test_split(fea_x, fea_y, test_size=0.8)

from sklearn.model_selection import learning_curve
n = [n for n in range(10, 46, 5)]
y1_train = []
y2_test = []

# for n in range(10, 46, 5):
for n in range(1, 17):
    # model = RandomForestClassifier(max_depth=n)
    # model = RandomForestClassifier(n_estimators=n)
    # model = RandomForestClassifier(max_depth=4, n_estimators=n)
    model = DecisionTreeClassifier(max_depth=n)
    train_sizes, train_score, test_score = learning_curve(model, fea_x, fea_y, train_sizes=[0.8], cv=2,
                                                          scoring='accuracy')
    # train_error =  1- np.mean(train_score,axis=1)
    # test_error = 1- np.mean(test_score,axis=1)
    y1 = np.mean(train_score, axis=1)
    # print(n, y1)
    y1_train.append(y1)
    y2 = np.mean(test_score, axis=1)
    # print(n, y2)
    y2_test.append(y2)

print(f"<Hyperparameter Analysis> End...")

##############################
# Training Module
##############################

print(f"<Training Module> Enter the best hyperparameters...")
# clf = RandomForestClassifier(n_estimators=20, max_depth=5)
clf = DecisionTreeClassifier(max_depth=4)
# clf = LogisticRegression()
# clf = GaussianNB()
# clf = KNeighborsClassifier(n_neighbors=16)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf = BallTree()

# clf = XGBClassifier()
# clf = LGBMClassifier(learning_rate=0.01, n_estimators=2000, n_jobs=2)

# Standardization标准化: 将特征数据的分布调整成标准正太分布，使得数据的均值维0，方差为1
# ss = StandardScaler()
# train_X = ss.fit_transform(train_df[features_to_use])
# test_X = ss.transform(test_df[features_to_use])

train_X = train_df[features_to_use].values
test_X = test_df[features_to_use].values




##############################
# Output Results
##############################

print(f"<Training Module> Start Cross-Validation...")
test_pred = None
log_loss_score = []
for train_idx, test_idx in model_selection.KFold(n_splits=10, shuffle=True, random_state=2020).split(train_X, train_df['interest_level']):
    clf.fit(train_X[train_idx], train_y[train_idx])
#   # 使用Light gbm
#     clf.fit(train_X[train_idx], train_y[train_idx],
#             eval_set=[(train_X[test_idx], train_y[test_idx]), (train_X[test_idx], train_y[test_idx])],
#             verbose=20,
#             early_stopping_rounds=50)
    print(f"<Training Module> Training Data...")
    loss = log_loss(train_y[test_idx], clf.predict_proba(train_X[test_idx]))
    print(f"<Training Module> Calulate Log_loss...: [Logloss] ---- {loss}")
    log_loss_score.append(loss)
    if test_pred is None:
        test_pred = clf.predict_proba(test_X)
    else:
        test_pred += clf.predict_proba(test_X)

test_pred /= 10

print()
print(f"<Training Module> [Final Result]>>>>>>[logloss]: {min(log_loss_score)}")
print(f"<Training Module> Output Data...")
out_df = pd.DataFrame(test_pred)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
# out_df.to_csv("random_forest_2.csv", index=False)






