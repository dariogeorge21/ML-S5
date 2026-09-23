import pandas as pd
import numpy as np

df = pd.read_excel("online-retail-dataset.xlsx")
print("Original Record:", len(df))

df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"]>0) & (df["UnitPrice"] > 0)]
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

