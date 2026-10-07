import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_excel("online-retail-dataset.xlsx")
print("Original Record:", len(df))

print(df.head())
print(df.info())

df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"]>0) & (df["UnitPrice"] > 0)]
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

print(df.shape)
print(df.head)

customer_data = df.groupby("CustomerID").agg(
   TotalSpend = ("TotalAmount", "sum"),
   TotalQuantity = ("Quantity", "sum"),
   NumberOfTransactions = ("InvoiceNo", "nunique"),
   NumberOfProducts = ("StockCode", "nunique"),
   AverageOrderValue = ("TotalAmount", "mean"),
   Country = ("Country", "first"),
).reset_index()

print(customer_data.head())

customer_data["Segment"] = pd.qcut(
    customer_data["TotalSpend"],
    q=3,
    labels=["Low","Medium","High"]
)

print("\nCustomer Segments:")
print(customer_data["Segment"].value_counts())

features = [
    "TotalQuantity",
    "NumberOfTransactions",
    "NumberOfProducts",
    "AverageOrderValue"
]

X = customer_data[features]
y = customer_data["Segment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

id3_model = DecisionTreeClassifier(
    criterion = "entropy",
    random_state = 42,
    max_depth = 4
)

id3_model.fit(X_train, y_train)
y_pred = id3_model.predict(X_test)

print("Predictions:")
print(y_pred[:20])

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

importance = pd.DataFrame({
    "Feature": features,
    "Importance": id3_model.feature_importances_
})

importance_sorted = importance.sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(22,12))
plot_tree(
    id3_model,
    feature_names=features,
    class_names=id3_model.classes_,
    filled=True,
    rounded=True,
    proportion = False
)

plt.barh(
    importance_sorted["Feature"],
    importance_sorted["Importance"]
)

plt.xlabel("Information-based importance")
plt.ylabel("Customer Behaviour Feature")
plt.title("Feature Importance in Customer Segmentation")

plt.show()