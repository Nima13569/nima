from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf1 = KNeighborsClassifier()
clf2 = DecisionTreeClassifier(random_state=0)
clf3 = LogisticRegression(max_iter=1000)

voting = VotingClassifier(
    estimators=[("knn", clf1), ("dt", clf2), ("lr", clf3)],
    voting="hard"
)
voting.fit(X_train, y_train)
print("Voting clf accuracy:", accuracy_score(y_test, voting.predict(X_test)))
#اینا توی vscode  خوب کار میکنه ولی توی
#jupyterlab نه