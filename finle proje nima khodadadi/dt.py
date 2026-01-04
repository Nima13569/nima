from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,  accuracy_score , mean_squared_error


wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


y_pred = dt.predict(X_test)


conf_matrix = confusion_matrix(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)


print("Confusion Matrix:")
print(conf_matrix)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Accuracy: {accuracy:.4f}")