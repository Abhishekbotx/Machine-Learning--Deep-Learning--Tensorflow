from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd;
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#--------LOGISTIC REGRESSION ---------

iris = load_iris()
X = iris.data
y = iris.target

lr_model = LogisticRegression(max_iter=500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model.fit(X_train, y_train)

pred = lr_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))


# --------KNN -----------

knn_model = KNeighborsClassifier(n_neighbors=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_model.fit(X_train, y_train)

pred = knn_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))


# FF --> Both KNN and LOGISTIC REGRESSION are Classification Problem Type

#Logistic regression example 2

data = {
    "marks": [20, 25, 35, 42, 50, 55, 60, 72, 85, 90],
}

df = pd.DataFrame(data)

# Convert marks into Pass/Fail
df["pass"] = df["marks"].apply(lambda x: 1 if x >=40 else 0)
print("df:",df)

x2=df[["marks"]]
y2=df["pass"]

lr2_model = LogisticRegression()
x2_train,x2_test,y2_train,y2_test=train_test_split(
    x2, y2, test_size=0.3, random_state=42
)


lr2_model.fit(x2_train,y2_train)

y2probs=lr2_model.predict_proba(x2_test)
print("\nPredicted Probabilities:\n", y2probs)
# Predicted Probabilities: 
# [[6.50e-14   1.00e+00],   → Almost certainly class 1 (pass)
# [9.9985e-01 1.49e-04],   → Almost certainly class 0 (fail) --> 99.985% → Fail 0.015% → Pass
# [2.08e-05   9.99979e-01]] → Almost certainly class 1 (pass)

# Predict classes
y2_pred = lr2_model.predict(x2_test)
print("\nPredicted Classes:", y2_pred)
# Predicted Classes: [1 0 1]
# It applied threshold:
# If PASS probability ≥ 0.4 → classify as PASS (1)
# Else classify FAIL (0)

# Evaluation
print("\nAccuracy:", accuracy_score(y2_test, y2_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y2_test, y2_pred))
print("\nClassification Report:\n", classification_report(y2_test, y2_pred))