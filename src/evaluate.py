from sklearn.metrics import r2_score, mean_squared_error
from train_model import model, X_test, y_test
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

#Calculate R² Score(1 → perfect model,0 → model explains nothing,<0 → very poor model)
print("R2 Score:", r2_score(y_test, y_pred))

#Calculate Mean Squared Error (Lower value = better model.)
print("MSE:", mean_squared_error(y_test, y_pred))

# Step 3: visualize predictions
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.show()
