# Importing Libraries
import warnings
from collections import Counter
import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Initial Configurations
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_rows", 100)
pd.set_option("display.precision", 3)
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Loading Dataset
df = pd.read_csv("./data/healthcare-dataset-stroke-data.csv")
df.head()

# Display the shape of the dataset
df.shape

# Display column information
print("Columns Information:")
df.info()

# Display basic statistics
print("Basic Stats:")
df.describe().T

# Handling Missing Values
# Display missing values
missing_values_count = df.isnull().sum()
total_cells = np.prod(df.shape)
total_missing_values = missing_values_count.sum()
percentage_missing = (total_missing_values / total_cells) * 100
percentage_missing_per_column = (missing_values_count / df.shape[0]) * 100

print("Missing Values:")
print(missing_values_count)
print(f"\nTotal Missing Values: {total_missing_values}")
print(f"Percentage of Missing Data: {percentage_missing:.3f} %")
print("\nPercentage of Missing Values Per Column:")
print(percentage_missing_per_column)

# Visualize missing values
msno.matrix(df, figsize=(10, 5), color=(0.2, 0.4, 0.6))

# Fill missing values in 'bmi' column with mean
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
msno.matrix(df, figsize=(10, 5), color=(0.2, 0.4, 0.6))

# Dropping Unnecessary Columns
df = df.drop(columns=['id'], errors='ignore')
df.head()

# Removing Duplicate Data
print("Total fully duplicated rows:", df.duplicated().sum())
duplicate_features = df.drop(columns=["stroke"]).duplicated().sum()
print(f"Rows with duplicate features but possibly different outcomes: {duplicate_features}")

# Converting Data Types
df['age'] = df['age'].astype(int)

# Encoding Categorical Variables
mdf = df.copy(deep=True)
categorical_features = []
discrete_features = []
binary_features = []
non_binary_features = []

for i in mdf.columns:
    if df[i].nunique() > 6:
        discrete_features.append(i)
    else:
        categorical_features.append(i)
        if df[i].nunique() == 2:
            binary_features.append(i)
        else:
            non_binary_features.append(i)

print('Categorical Features :', *categorical_features)
print('Discrete Features :', *discrete_features)
print('Binary Categorical Features:', *binary_features)
print('Non_Binary Categorical Features', *non_binary_features)

# Label Encoding for binary features
le = LabelEncoder()
binary_features = [col for col in binary_features if col not in ['hypertension', 'heart_disease']]
print("Encoding Binary Features:")
for col in binary_features:
    mdf[col] = le.fit_transform(mdf[col])
    print(f"{col} : {mdf[col].unique()} = {le.inverse_transform(mdf[col].unique())}")

# One-Hot Encoding for non-binary features
mdf['gender'] = pd.Categorical(
    mdf['gender'],
    categories=['Other', 'Male', 'Female'],
    ordered=False
)
mdf = pd.get_dummies(mdf, columns=['gender'], drop_first=True, dtype=int)
mdf.head()

# Creating New Features
mdf['has_smoked'] = mdf['smoking_status'].apply(
    lambda x: 1 if x in ['formerly smoked', 'smokes'] else 0 if x == 'never smoked' else None
)
mdf['is_working'] = mdf['work_type'].apply(
    lambda x: 1 if x in ['Private', 'Self_employed', 'Govt_job', 'children'] else 0 if x == 'Never_worked' else None
)

mdf['has_smoked'] = mdf['has_smoked'].fillna(mdf['has_smoked'].mode()[0])
mdf['is_working'] = mdf['is_working'].fillna(mdf['is_working'].mode()[0])
mdf['is_working'] = mdf['is_working'].astype(int)
mdf['has_smoked'] = mdf['has_smoked'].astype(int)

mdf.drop(columns=['smoking_status', 'work_type'], inplace=True, errors='ignore')
categorical_features.clear()

# Renaming Columns
mdf = mdf.rename(columns={
    'gender_Male': 'Male',
    'gender_Female': 'Female',
    'hypertension': 'Has_Hypertension',
    'heart_disease': 'Has_Heart_Disease',
    'avg_glucose_level': 'Avg_Glucose',
    'bmi': 'Body_Mass_Index',
    'stroke': 'Stroke_Chance',
    'age': 'Age',
    'has_smoked': 'Has_Smoked',
    'is_working': 'Is_Working',
    'gender': 'Gender',
    'ever_married': 'Married',
    'Residence_type': 'Residence_Type'
})
mdf = mdf[['Age', 'Male', 'Female', 'Married', 'Residence_Type', 'Has_Hypertension', 'Has_Heart_Disease', 'Has_Smoked', 'Is_Working',
           'Body_Mass_Index', 'Avg_Glucose', 'Stroke_Chance']]

categorical_features = [col for col in mdf.columns if len(mdf[col].unique()) <= 2 and col not in ['Stroke_Chance']]
discrete_features.clear()
discrete_features = [col for col in mdf.columns if len(mdf[col].unique()) > 6]
print(f"Current Categorical Features: {categorical_features}")
print(f"Current Discrete Features: {discrete_features}")
print("Current Dataset:")
mdf.head()

# Handling Outliers
numeric_cols = ['Age', 'Body_Mass_Index', 'Avg_Glucose']

# Plot boxplots
plt.figure(figsize=(18, 12))
for i, col in enumerate(numeric_cols):
    plt.subplot(4, 4, i + 1)
    sns.boxplot(y=mdf[col], color='lightblue')
    plt.title(col)
    plt.tight_layout()

plt.show()

# Remove outliers using IQR
def remove_outliers_iqr(mdf, column):
    Q1 = mdf[column].quantile(0.25)
    Q3 = mdf[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 0.5 * IQR
    upper_bound = Q3 + 0.5 * IQR
    return mdf[(mdf[column] >= lower_bound) & (mdf[column] <= upper_bound)]

mdf = remove_outliers_iqr(mdf, 'Body_Mass_Index')
mdf = remove_outliers_iqr(mdf, 'Avg_Glucose')

# Data Balancing
over_sampling = SMOTE(sampling_strategy=1)
under_sampling = RandomUnderSampler(sampling_strategy=0.1)
features = mdf.loc[:, :'Avg_Glucose']
target = mdf.loc[:, 'Stroke_Chance']

steps = [('under', under_sampling), ('over', over_sampling)]
pipeline = Pipeline(steps=steps)
features, target = pipeline.fit_resample(features, target)

Counter(target)

# Data Splitting
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=RANDOM_STATE)

# Feature Scaling
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

discrete_features = [col for col in discrete_features if col not in ['Avg_Glucose']]
categorical_features = [col for col in categorical_features if col not in ['Has_Heart_Disease', 'Has_Hypertension']]

for col in discrete_features:
    if col in x_train.columns and col in x_test.columns:
        x_train[[col]] = minmax_scaler.fit_transform(x_train[[col]])
        x_test[[col]] = minmax_scaler.transform(x_test[[col]])

for col in categorical_features:
    if col in x_train.columns and col in x_test.columns:
        x_train[[col]] = standard_scaler.fit_transform(x_train[[col]])
        x_test[[col]] = standard_scaler.transform(x_test[[col]])

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='accuracy', cv=5, n_jobs=-1, verbose=2)

grid_search.fit(x_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(x_test)

# Model Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_tuned):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_tuned):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred_tuned):.2f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_tuned):.2f}")

cm = confusion_matrix(y_test, y_pred_tuned)
total_samples = np.sum(cm)

# Calculate percentages
cm_percent = (cm / cm.sum(axis=1)[:, np.newaxis]) * 100  # Row-wise percentage

class_names = ['No Stroke', 'Stroke']

annot = np.empty_like(cm, dtype=object)
annot[0, 0] = f"TN\n{cm[0,0]}\n({cm_percent[0,0]:.1f}%)"
annot[0, 1] = f"FP\n{cm[0,1]}\n({cm_percent[0,1]:.1f}%)"
annot[1, 0] = f"FN\n{cm[1,0]}\n({cm_percent[1,0]:.1f}%)"
annot[1, 1] = f"TP\n{cm[1,1]}\n({cm_percent[1,1]:.1f}%)"

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm,
            annot=annot,
            fmt='',  # Disable default annotation
            cmap='Blues',
            linewidths=0.5,
            linecolor='black',
            cbar=False,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix\n(TP, FP, TN, FN with Counts & Percentages)')
plt.show()