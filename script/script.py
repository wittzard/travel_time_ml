import pandas as pd
import numpy as np
import joblib
import os
# =========================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

cwd = os.getcwd()
# =========================================================================
data_folder = os.path.join(cwd, 'data')  

file_name = 'Food_Delivery_Times.csv'
file_path = os.path.join(data_folder, file_name)

if os.path.exists(file_path):
    print(f"File found: {file_path}")
    df = pd.read_csv(file_path)
else:
    raise FileNotFoundError(f"File not found: {file_path}")
    
# =========================================================================

# clean and tranform data
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.drop(columns=['order_id'])

categorical_cols = ['weather','traffic_level','time_of_day','vehicle_type']
numeric_cols = ['distance_km','preparation_time_min','courier_experience_yrs','delivery_time_min']

df[categorical_cols] = df[categorical_cols].apply(lambda x: x.str.lower().str.replace(' ', '_'))

df = df.dropna()

# split data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# y Final ==================================================================
y_train = df_train['delivery_time_min'].values
y_val = df_val['delivery_time_min'].values
y_test = df_test['delivery_time_min'].values
# ==========================================================================

del df_train['delivery_time_min']
del df_val['delivery_time_min']
del df_test['delivery_time_min']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

categorical_cols = ['weather','traffic_level','time_of_day','vehicle_type']

encoder.fit(df_train[categorical_cols])

X_train_cat = encoder.transform(df_train[categorical_cols])
X_val_cat = encoder.transform(df_val[categorical_cols])
X_test_cat = encoder.transform(df_test[categorical_cols])

numeric_cols = [col for col in df_train.columns if col not in categorical_cols]

# X Final ==================================================================
X_train = np.hstack([df_train[numeric_cols].values, X_train_cat])
X_val = np.hstack([df_val[numeric_cols].values, X_val_cat])
X_test = np.hstack([df_test[numeric_cols].values, X_test_cat])
# ==========================================================================

model_folder = os.path.join(cwd, 'model')  

encoder_file_name = 'encoder.pkl'
encoder_file_path = os.path.join(model_folder, encoder_file_name)

# use encoder for easy to encode on production
joblib.dump(encoder, encoder_file_path)
print(f"encoder saved to: {encoder_file_path}")

# for tracking models
results = []

# ===========================================================================
# Model Trianing choose best MSE 
# ===========================================================================
lin_model = LinearRegression()

lin_model.fit(X_train, y_train)

y_pred_val = lin_model.predict(X_val)
y_pred_test = lin_model.predict(X_test)

mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

results.append({
    "Model": "linear_model",
    "Model Object": lin_model,
    "Validation MSE": mse_val,
    "Validation MAE": mae_val,
    "Test MSE": mse_test,
    "Test MAE": mae_test
})
# ============================================================================
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

base_model = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
rf_best_model = grid_search.best_estimator_

y_pred_val = rf_best_model.predict(X_val)
y_pred_test = rf_best_model.predict(X_test)

mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

results.append({
    "Model": "random_forest_regressor",
    "Model Object": rf_best_model,
    "Validation MSE": mse_val,
    "Validation MAE": mae_val,
    "Test MSE": mse_test,
    "Test MAE": mae_test
})
# ============================================================================
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 300],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

# Base model for grid search
base_model = xgb.XGBRegressor(
    random_state=42,
    eval_metric="rmse"
)

# Perform grid search
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
xgb_best_model = grid_search.best_estimator_

y_pred_val = xgb_best_model.predict(X_val)
y_pred_test = xgb_best_model.predict(X_test)

mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

results.append({
    "Model": "xgb_regressor",
    "Model Object": xgb_best_model,
    "Validation MSE": mse_val,
    "Validation MAE": mae_val,
    "Test MSE": mse_test,
    "Test MAE": mae_test
})
# ============================================================================
df_results = pd.DataFrame(results)

if 'Model Object' in df_results.columns:
    df_results = df_results.drop(columns=['Model Object'])

print(df_results)

best_result = min(results, key=lambda x: x["Validation MSE"])
best_model = best_result["Model Object"]

# ============================================================================
model_folder = os.path.join(cwd, 'model')  

model_file_name = 'best_model.pkl'
model_file_path = os.path.join(model_folder, model_file_name)

joblib.dump(best_model, model_file_path)
print(f"Model saved to: {model_file_path}")
# ============================================================================