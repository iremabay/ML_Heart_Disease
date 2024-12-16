import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#Veri Yükleme ve İlk İnceleme
def load_and_inspect_data(filepath):
    df = pd.read_csv(filepath)
    print("Veri Seti İlk 5 Satır:")
    print(df.head())
    print("\nVeri Seti Bilgisi:")
    print(df.info())
    print("\nEksik Değer Kontrolü:")
    print(df.isnull().sum())
    print(df.dtypes)
    return df

df = load_and_inspect_data("data/heart_2020_cleaned.csv")

#Değişken Türlerini Belirleme
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtype != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtype == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O" and col not in num_but_cat]
    print(f"Kategorik Değişkenler: {len(cat_cols)} | Sayısal Değişkenler: {len(num_cols)}")
    return cat_cols, num_cols

cat_cols, num_cols = grab_col_names(df)

#Hedef Değişken Analizi
def target_summary(dataframe, target):
    print("\nHedef Değişken Sınıf Dağılımı:")
    print(dataframe[target].value_counts(normalize=True) * 100)

target_summary(df, "HeartDisease")

# Özellik Mühendisliği
def feature_engineering(dataframe):
    age_mapping = {'80 or older': 85, '75-79': 77, '70-74': 72, '65-69': 67,
                   '60-64': 62, '55-59': 57, '50-54': 52, '45-49': 47,
                   '40-44': 42, '35-39': 37, '30-34': 32, '25-29': 27, '18-24': 21}

    # Yaş kategorilerini sayısal hale getirme ve eksik değerleri doldurma
    dataframe['AgeCategory'] = dataframe['AgeCategory'].map(age_mapping)
    if dataframe['AgeCategory'].isnull().any():
        dataframe['AgeCategory'] = dataframe['AgeCategory'].fillna(np.median(list(age_mapping.values())))

    # Sağlık risk skoru
    dataframe['Risk_Score'] = dataframe['Smoking'].map({1: 1, 0: 0}) + \
                              dataframe['AlcoholDrinking'].map({1: 1, 0: 0}) + \
                              (1 - dataframe['PhysicalActivity'].map({1: 1, 0: 0})) + \
                              (dataframe['BMI'] / 10)

    # Risk_Score'da NaN değerleri doldur
    dataframe['Risk_Score'] = dataframe['Risk_Score'].fillna(dataframe['Risk_Score'].median())

    # Uykuyu kategorize etme
    dataframe['SleepCategory'] = pd.cut(dataframe['SleepTime'], bins=[0, 5, 9, 24],
                                        labels=['Low', 'Normal', 'High'])

    print("\nYeni Değişkenler Eklendi.")
    return dataframe


# Kategorik Değişkenlerin Label Encoding ile Sayısal Hale Getirilmesi
def label_encode(dataframe, cat_columns):

    le = LabelEncoder()
    for col in cat_columns:
        dataframe[col] = le.fit_transform(dataframe[col].astype(str))
    print("\nKategorik Değişkenler Encode Edildi.")
    return dataframe

def fix_categorical_columns(df):
    """Verideki tüm kategorik sütunları sayısal hale getirir."""
    cat_cols, _ = grab_col_names(df)
    df = label_encode(df, cat_cols)
    return df

df = fix_categorical_columns(df)
df = feature_engineering(df)

print("AgeCategory'deki eşleşmeyen kategoriler:")
print(df['AgeCategory'].unique())

# Smoking, AlcoholDrinking, PhysicalActivity sütunlarındaki hatalı değerleri kontrol et
print("\nSmoking değerleri:")
print(df['Smoking'].unique())
print("\nAlcoholDrinking değerleri:")
print(df['AlcoholDrinking'].unique())
print("\nPhysicalActivity değerleri:")
print(df['PhysicalActivity'].unique())
print("\nBMI sütununda hatalı değer kontrolü:")
print(df['BMI'].dtype)


# Özellik mühendisliği sonrası NaN kontrolü
print("\nYeni değişkenlerde NaN kontrolü:")
print(df[['AgeCategory', 'Risk_Score', 'SleepCategory']].isnull().sum())


#SMOTE ile Dengesiz Veri Düzeltme
def apply_smote(X_train, y_train):
    """SMOTE uygulaması ile eğitim verisini dengeleme."""
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res


from sklearn.model_selection import GridSearchCV


# def hyperparameter_tuning(X_train_res, y_train_res):
#     """Random Forest için hiperparametre optimizasyonu."""
#     param_grid = {
#         'n_estimators': [100, 200],
#         'max_depth': [10, 20, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }
#
#     grid_search = GridSearchCV(
#         estimator=RandomForestClassifier(random_state=42),
#         param_grid=param_grid,
#         scoring='f1',
#         cv=3,
#         n_jobs=-1,
#         verbose=2
#     )
#
#     grid_search.fit(X_train_res, y_train_res)
#     print("En İyi Parametreler:", grid_search.best_params_)
#     return grid_search.best_estimator_


def train_and_evaluate_RandomForest(dataframe, target_col):
    y = dataframe[target_col]
    X = dataframe.drop(target_col, axis=1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SMOTE işlemi için yalnızca sayısal sütunları seç
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train_numeric = X_train[numeric_cols]
    X_test_numeric = X_test[numeric_cols]

    # SMOTE ile eğitim verisini dengeleme
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_numeric, y_train)

    # Random Forest modeli
    model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)  # Varsayılan parametreler
    model.fit(X_train_res, y_train_res)
    y_pred_res = model.predict(X_test_numeric)


    print("\nModel Performansı (Random Forest):")
    print(f"Doğruluk (Accuracy): {accuracy_score(y_test, y_pred_res):.3f}")
    print(f"Duyarlılık (Recall): {recall_score(y_test, y_pred_res):.3f}")
    print(f"Kesinlik (Precision): {precision_score(y_test, y_pred_res):.3f}")
    print(f"F1 Skoru: {f1_score(y_test, y_pred_res):.3f}")
    print(f"AUC Skoru: {roc_auc_score(y_test, y_pred_res):.3f}")

    feature_importance = pd.DataFrame({'Değişken': numeric_cols, 'Önem': model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Önem', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Önem', y='Değişken', data=feature_importance)
    plt.title("Değişken Önem Düzeyleri")
    plt.show()


from xgboost import XGBClassifier

def train_and_evaluate_xgboost(dataframe, target_col):
    y = dataframe[target_col]
    X = dataframe.drop(target_col, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SMOTE işlemi için yalnızca sayısal sütunları seç
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train_numeric = X_train[numeric_cols]
    X_test_numeric = X_test[numeric_cols]

    # SMOTE ile eğitim verisini dengeleme
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_numeric, y_train)

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )

    model.fit(X_train_res, y_train_res)
    y_pred_res = model.predict(X_test_numeric)

    print("\nModel Performansı (XGBoost):")
    print(f"Doğruluk (Accuracy): {accuracy_score(y_test, y_pred_res):.3f}")
    print(f"Duyarlılık (Recall): {recall_score(y_test, y_pred_res):.3f}")
    print(f"Kesinlik (Precision): {precision_score(y_test, y_pred_res):.3f}")
    print(f"F1 Skoru: {f1_score(y_test, y_pred_res):.3f}")
    print(f"AUC Skoru: {roc_auc_score(y_test, y_pred_res):.3f}")


def train_and_evaluate_KNN(dataframe, target_col):
    y = dataframe[target_col]
    X = dataframe.drop(target_col, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train_numeric = X_train[numeric_cols]
    X_test_numeric = X_test[numeric_cols]

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_numeric, y_train)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_res, y_train_res)
    y_pred_res = model.predict(X_test_numeric)

    print("\nModel Performansı (KNN):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_res):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred_res):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred_res):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_res):.3f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_res):.3f}")


def train_and_evaluate_LogisticRegression(dataframe, target_col):
    y = dataframe[target_col]
    X = dataframe.drop(target_col, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train_numeric = X_train[numeric_cols]
    X_test_numeric = X_test[numeric_cols]

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_numeric, y_train)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_res, y_train_res)
    y_pred_res = model.predict(X_test_numeric)

    print("\nModel Performansı (Logistic Regression):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_res):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred_res):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred_res):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_res):.3f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_res):.3f}")

def train_and_evaluate_SVC(dataframe, target_col):
    y = dataframe[target_col]
    X = dataframe.drop(target_col, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train_numeric = X_train[numeric_cols]
    X_test_numeric = X_test[numeric_cols]

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_numeric, y_train)

    model = SVC(probability=True, random_state=42)
    model.fit(X_train_res, y_train_res)
    y_pred_res = model.predict(X_test_numeric)

    print("\nModel Performansı (SVC):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_res):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred_res):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred_res):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_res):.3f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_res):.3f}")


train_and_evaluate_RandomForest(df, "HeartDisease")
train_and_evaluate_LogisticRegression(df, "HeartDisease")
train_and_evaluate_KNN(df, "HeartDisease")
train_and_evaluate_xgboost(df, "HeartDisease")
train_and_evaluate_SVC(df, "HeartDisease")