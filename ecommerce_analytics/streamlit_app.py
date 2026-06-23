import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="E-commerce Analytics (RFM + Clustering + Recs)", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data(show_spinner=False)
def load_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    for col in ["Quantity", "UnitPrice"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(str)

    # Remove returns/invalid rows
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] >= 0)].copy()
    df["Amount"] = df["Quantity"] * df["UnitPrice"]

    df = df.dropna(subset=["InvoiceDate"])
    return df


def rfm_scoring(clean_df: pd.DataFrame) -> pd.DataFrame:
    max_date = clean_df["InvoiceDate"].max()

    cust = clean_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (max_date - s.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Amount", "sum"),
    ).reset_index()

    def qcut_series(s, q=4, reverse=False):
        # Recency lower is better (reverse=True), others higher is better
        if s.nunique() < q:
            r = s.rank(method="average")
            bins = pd.qcut(r, q=q, labels=False, duplicates="drop")
            out = (bins.astype(float) + 1).astype(int)
        else:
            bins = pd.qcut(s, q=q, labels=False, duplicates="drop")
            out = (bins.astype(float) + 1).astype(int)

        if reverse:
            out = (q + 1 - out)
        return out

    cust["R_Score"] = qcut_series(cust["Recency"], q=4, reverse=True)
    cust["F_Score"] = qcut_series(cust["Frequency"], q=4, reverse=False)
    cust["M_Score"] = qcut_series(cust["Monetary"], q=4, reverse=False)

    cust["RFM_Score"] = cust["R_Score"].astype(str) + cust["F_Score"].astype(str) + cust["M_Score"].astype(str)
    cust["RFM_Score_num"] = cust["R_Score"] + cust["F_Score"] + cust["M_Score"]
    return cust


def cluster_customers(clean_df: pd.DataFrame, k=6):
    cust = clean_df.groupby("CustomerID").agg(
        recency=("InvoiceDate", lambda s: (clean_df["InvoiceDate"].max() - s.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("Amount", "sum"),
    ).reset_index()

    X = cust[["recency", "frequency", "monetary"]].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    cust["cluster"] = km.fit_predict(Xs)

    # PCA for visualization
    X2 = PCA(n_components=2, random_state=42).fit_transform(Xs)
    cust["pca1"] = X2[:, 0]
    cust["pca2"] = X2[:, 1]
    return cust


def make_customer_item_matrix(clean_df: pd.DataFrame, min_item_support=25):
    clean = clean_df.copy()

    if "StockCode" in clean.columns:
        item_col = "StockCode"
    elif "Description" in clean.columns:
        item_col = "Description"
    else:
        raise ValueError("Missing both StockCode and Description; cannot build recommendations.")

    clean[item_col] = clean[item_col].astype(str)

    item_support = clean.groupby(item_col)["CustomerID"].nunique()
    frequent_items = item_support[item_support >= min_item_support].index
    clean = clean[clean[item_col].isin(frequent_items)]

    pivot = clean.pivot_table(
        index="CustomerID",
        columns=item_col,
        values="InvoiceNo",
        aggfunc="nunique",
        fill_value=0
    )

    likes = (pivot > 0).astype(np.uint8)  # implicit: purchased or not
    return likes, item_col


def recommend_by_customer_similarity(likes: pd.DataFrame, customer_id: str, top_k=10, min_common=2):
    if customer_id not in likes.index:
        return pd.DataFrame({"Item": [], "Score": []})

    # customer like vectors
    target_vec = likes.loc[customer_id].values.astype(bool)
    all_vecs = likes.values.astype(bool)

    intersection = (all_vecs & target_vec).sum(axis=1)
    union = (all_vecs | target_vec).sum(axis=1)

    sim = np.zeros(likes.shape[0], dtype=float)
    mask = union > 0
    sim[mask] = intersection[mask] / union[mask]

    # Require some overlap
    valid = intersection >= min_common
    sim = sim * valid.astype(float)

    # Weighted popularity of items among similar customers
    weighted = likes.values.T @ sim

    already = likes.loc[customer_id].astype(bool).values
    weighted[already] = -np.inf

    top_idx = np.argsort(weighted)[::-1][:top_k]
    top_idx = top_idx[np.isfinite(weighted[top_idx])]

    items = likes.columns[top_idx]
    scores = weighted[top_idx]

    return pd.DataFrame({"Item": items, "Score": scores}).sort_values("Score", ascending=False).reset_index(drop=True)


def main():
    st.title("E-commerce Analytics: EDA + RFM + KMeans + Recommendations")

    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader("Upload online_retail.csv", type=["csv"])

        st.header("Parameters")
        min_item_support = st.slider("Min item support (customers)", 5, 200, 25, step=1)
        k_clusters = st.slider("KMeans clusters (customers)", 2, 12, 6, step=1)
        top_k_recs = st.slider("Top recommendations", 5, 30, 10, step=1)
        min_common = st.slider("Min common items between customers", 1, 10, 2, step=1)

    if uploaded is None:
        st.info("Upload a CSV to begin.")
        return

    try:
        df = pd.read_csv(uploaded)
        clean_df = load_and_clean(df)
    except Exception as e:
        st.error(f"Failed to process CSV: {e}")
        return

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Quick EDA (after cleaning)")
        st.write(f"Rows: {len(clean_df):,}")
        st.write(f"Customers: {clean_df['CustomerID'].nunique():,}")
        st.write(f"Countries (if present): {clean_df['Country'].nunique() if 'Country' in clean_df.columns else 'N/A'}")
        st.write(f"Date range: {clean_df['InvoiceDate'].min().date()} → {clean_df['InvoiceDate'].max().date()}")

    with c2:
        st.subheader("Revenue by month")
        tmp = clean_df.copy()
        tmp["month"] = tmp["InvoiceDate"].dt.to_period("M").astype(str)
        rev = tmp.groupby("month")["Amount"].sum().reset_index().sort_values("month")

        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.lineplot(data=rev.tail(24), x="month", y="Amount", marker="o", ax=ax)
        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue (sum Quantity*UnitPrice)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig, clear_figure=True)

    st.divider()

    st.subheader("RFM Scoring")
    cust_rfm = rfm_scoring(clean_df)

    st.dataframe(cust_rfm.sort_values("RFM_Score_num", ascending=False).head(50), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        sns.histplot(cust_rfm["RFM_Score_num"], bins=15, ax=ax)
        ax.set_xlabel("R+F+M score (1..12)")
        ax.set_ylabel("Customers")
        st.pyplot(fig, clear_figure=True)

    with c4:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        rfm_top = cust_rfm["RFM_Score"].value_counts().head(10)
        sns.barplot(x=rfm_top.index, y=rfm_top.values, ax=ax)
        ax.set_xlabel("RFM_Score (R,F,M)")
        ax.set_ylabel("Customers")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig, clear_figure=True)

    st.divider()

    st.subheader("Customer Clustering (KMeans)")
    cust_cluster = cluster_customers(clean_df, k=k_clusters)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=cust_cluster,
        x="pca1",
        y="pca2",
        hue="cluster",
        palette="tab10",
        s=35,
        ax=ax
    )
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig, clear_figure=True)

    st.write("Cluster summary:")
    st.dataframe(
        cust_cluster.groupby("cluster")
        .agg(customers=("CustomerID", "nunique"),
             recency=("recency", "mean"),
             frequency=("frequency", "mean"),
             monetary=("monetary", "mean"))
        .reset_index()
        .sort_values("monetary", ascending=False),
        use_container_width=True
    )

    st.divider()

    st.subheader("Recommendations (Collaborative Filtering-style)")
    try:
        likes, _item_col = make_customer_item_matrix(clean_df, min_item_support=min_item_support)
    except Exception as e:
        st.error(f"Cannot build recommendations: {e}")
        return

    cust_ids = likes.index.tolist()
    # keep UI manageable
    options = cust_ids[:2000] if len(cust_ids) > 2000 else cust_ids

    with st.form("recommend_form"):
        customer_id = st.selectbox("Select customer", options)
        submitted = st.form_submit_button("Recommend")

    if submitted:
        recs = recommend_by_customer_similarity(likes, customer_id=customer_id, top_k=top_k_recs, min_common=min_common)
        if recs.empty:
            st.warning("No recommendations found for this customer with the current settings.")
        else:
            st.dataframe(recs, use_container_width=True)

            fig, ax = plt.subplots(figsize=(7.5, 3.5))
            sns.barplot(data=recs.head(10), x="Item", y="Score", ax=ax)
            ax.set_xlabel("Item")
            ax.set_ylabel("Similarity-based score")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig, clear_figure=True)


if __name__ == "__main__":
    main()
