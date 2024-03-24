import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
sales_data = pd.read_csv(r"C:\\Users\Bhavani\OneDrive\Desktop\durga\Project\advertising.csv")
print(sales_data.isnull())
print(sales_data.size)
print(sales_data.info())
print(sales_data.describe())
sns.pairplot(sales_data)
plt.show()
model = LinearRegression()
X = sales_data.drop(columns=['Sales'])
y = sales_data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
new_data_predictions = model.predict(X_test)
print("Forecasted Sales for New Data:")
for i, prediction in enumerate(new_data_predictions):
    print(f"Data Point {i+1}: {prediction}")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, color='red', label='Ideal Prediction')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.show()
param_grid = {
    'fit_intercept': [True, False],
    'positive': [True, False]
}
grid_search = GridSearchCV(estimator=LinearRegression(), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_model = LinearRegression(fit_intercept=best_params['fit_intercept'], positive=best_params['positive'])
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
print('Best Model Mean Squared Error:', mse_best)
