import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

data = pd.read_csv('dataset/StudentScore.xls', delimiter=',')

target = 'writing score'
X = data.drop(target, axis= 1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1)

num = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=-1, strategy='median')),
    ('trans', StandardScaler())
])

education_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree",
                    "master's degree"]
gender_values = ["male", "female"]
lunch_values = X_train['lunch'].unique()
test_values = X_train['test preparation course'].unique()
ordinal = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))
])

nomninal = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('trans', OneHotEncoder(sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num, ["reading score", "math score"]),
    ("ord_feature", ordinal, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_feature", nomninal, ["race/ethnicity"])
])

reg = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('linear', LinearRegression())
])

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
for i, j in zip(y_test, y_pred):
    print(i, ' new ', j)

print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
