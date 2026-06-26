# E-commerce Analytics (Streamlit): RFM + K-Means Clustering + Recommendations

This project is a Streamlit web app that lets you explore and model an **online retail** dataset using:

- **RFM scoring** (Recency, Frequency, Monetary).....
- **Customer segmentation** with **K-Means**
- **Item recommendations** based on customer similarity (implicit “purchased/not purchased” vectors)

## Features

1. Upload `online_retail.csv`
2. Quick EDA after cleaning:
   - rows / customers
   - date range
   - revenue trend by month
3. RFM scoring table and distributions
4. Customer clustering visualization (PCA 2D)
5. Interactive recommendations for a selected customer

## Project Structure

- `streamlit_app.py` — main Streamlit application
- `requirements.txt` — Python dependencies
- `notebooks/eda_rfm_clustering_recommender.ipynb` — exploratory notebook (optional)

## How to Run Locally

### 1) Create a virtual environment

```bash
python -m venv .venv
```

### 2) Install dependencies

```bash
# from this folder (ecommerce_analytics)
.venv\Scripts\python -m pip install -r requirements.txt
```

### 3) Start the Streamlit server

```bash
.venv\Scripts\python -m streamlit run streamlit_app.py --server.port 8501
```

Then open:

- `http://localhost:8501`

## Notes / Assumptions

- Input file must contain these columns (typical for UCI Online Retail):
  - `CustomerID`, `InvoiceNo`, `InvoiceDate`, `Quantity`, `UnitPrice`
- If `StockCode` exists it’s used as the item identifier; otherwise `Description` is used.

## Recommendations Logic (High Level)

- Create a customer-by-item matrix where a value = whether the customer purchased the item.
- For a selected customer, compute similarity with other customers using **Jaccard similarity** over liked items.
- Score items by weighted popularity among similar customers.
- Exclude items already purchased by the customer.

## Demo Checklist

- Run the app
- Upload the dataset
- Show:
  1) Revenue by month
  2) Top RFM customers
  3) Clustering scatter plot
  4) Recommendations for 1–2 customers

