from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import json

# 1️⃣ Load dataset
data = load_iris()
print("data::",data)
X = data.data       # features (measurements)sepal length, sepal width, petal length, petal width
y = data.target     # labels (flower species) encoded in 0,1,2 indexing 



# 2️⃣ Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 4️⃣ Predict
y_pred = model.predict(X_test)

# 5️⃣ Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
