from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,auc,confusion_matrix,r2_score
import numpy as np
import pandas as pd

marks = np.array([20, 25, 35, 40,42, 50, 55, 60, 72, 85, 90])
labels = np.array([0, 0, 0, 0, 0,1, 1, 1, 1, 1, 1])

# Create dataframe
df = pd.DataFrame({
    "Marks": marks,
    "Pass": labels
})

x = df[["Marks"]]
y = df["Pass"]

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.3,random_state=42
)



log_model=LogisticRegression()


dt_model=DecisionTreeClassifier()


knn_model=KNeighborsClassifier()


models=[log_model,dt_model,knn_model]

for model in models:
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    print("\nmodel:",str(model)[:-2])
    print("predicted classes::",y_pred)
    print("Predicted Probabilities:\n", y_prob)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("scores:", cross_val_score(model, x, y, cv=4))
     
    # print("Marks for 7 hours study:", model.predict([[7]])[0])
    # print("accuracy score:",accuracy_score(y_test,y_pred))
    # print("confusion matrix:",confusion_matrix(y_test,y_pred))
    