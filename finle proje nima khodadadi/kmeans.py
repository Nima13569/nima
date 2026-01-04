from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features 
y = wine_quality.data.targets 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)


labels = kmeans.labels_
print(labels)