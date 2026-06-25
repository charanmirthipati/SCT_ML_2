from flask import Flask, render_template
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)

df = pd.read_csv("Mall_Customers.csv")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

@app.route("/")
def home():
    total_customers = len(df)
    avg_income = round(df['Annual Income (k$)'].mean(), 2)
    avg_spending = round(df['Spending Score (1-100)'].mean(), 2)

    return render_template(
        "index.html",
        total_customers=total_customers,
        avg_income=avg_income,
        avg_spending=avg_spending
    )

if __name__ == "__main__":
    app.run(debug=True)
