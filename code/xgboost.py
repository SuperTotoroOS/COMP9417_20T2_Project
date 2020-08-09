import random
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from jsonschema._utils import uniq
from scipy import sparse
from warnings import filterwarnings
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import xgboost as xgb

filterwarnings('ignore')

train_df = pd.read_json('../store/train.json.zip', compression='zip')
test_df = pd.read_json('../store/test.json.zip', compression='zip')

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

# define XGBoost
def runXGB(train_X, train_y, val_X, val_y=None, test_X=None, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eval_metric'] = "mlogloss"
    param['eta'] = 0.3
    param['max_depth'] = 3
    param['min_child_weight'] = 4
    param['gamma'] = 0
    param['subsample'] = 0.75
    param['colsample_bytree'] = 0.75

    param['silent'] = 0
    param['num_class'] = 3

    param['seed'] = 0
    param['nthread'] = 12
    plst = list(param.items())

    num_rounds = num_rounds

    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if val_y is not None:
        xgval = xgb.DMatrix(val_X, label=val_y)
        watchlist = [(xgtrain, 'train'), (xgval, 'val')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
    else:
        model = xgb.train(plst, xgtrain, num_rounds)
    xgtest = xgb.DMatrix(test_X)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high': 2, 'medium': 1, 'low': 0}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

# train module
test_pred = None
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2020)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
    dev_X, val_X = train_X[dev_index, :], train_X[val_index, :]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    preds, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, num_rounds=2000)

    if test_pred is None:
        test_pred = preds
    else:
        test_pred += preds
test_pred /= 5
out_df = pd.DataFrame(test_pred)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("xgb.csv", index=False)

print(f"<Feature Engineering> Finished Feature Engineering: train_set{train_df.shape}")
print(f"<Feature Engineering> Finished Feature Engineering: test_set{test_df.shape}")
