from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# دریافت دیتاست Wine Quality
wine_quality = fetch_ucirepo(id=186)

# ویژگی‌ها و target
X = wine_quality.data.features
y = wine_quality.data.targets.values.ravel()  # تبدیل به آرایه 1D

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# تعریف مدل‌های پایه
# چون ویژگی‌ها مقیاس متفاوتی دارند، بهتر است از StandardScaler استفاده کنیم
clf1 = make_pipeline(StandardScaler(), KNeighborsClassifier())
clf2 = DecisionTreeClassifier(random_state=0)
clf3 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

# Voting Classifier (hard voting)
voting = VotingClassifier(
    estimators=[
        ("knn", clf1),
        ("dt", clf2),
        ("lr", clf3)
    ],
    voting="hard"
)

# آموزش و پیش‌بینی
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)

# دقت
accuracy = accuracy_score(y_test, y_pred)
print(f"Voting Classifier Accuracy on Wine Quality: {accuracy:.12f}")
#اینو از هوش مصنوعی گرفتم
#اینا توی vscode  خوب کار میکنه ولی توی
#jupyterlab نه