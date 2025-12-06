import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils import resample
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ----------------------------
# BASIC UI STYLE (lebih simple)
# ----------------------------
st.set_page_config(page_title="Analisis Apriori", layout="wide")

st.markdown("""
<style>
body {
    background-color: #f6f8fa;
}
.main > div {
    background-color: white;
    padding: 20px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("Analisis Association Rule - Apriori")
st.write("Upload dataset transaksi dan jalankan metode Apriori untuk menemukan pola pembelian konsumen.")

# ----------------------------
# UPLOAD FILE
# ----------------------------
uploaded = st.file_uploader("Pilih File Dataset (.xlsx / .csv)", type=["xlsx", "csv"])

if uploaded:
    if uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    # ----------------------------
    # DATA DASAR
    # ----------------------------
    st.subheader("Ringkasan Data")
    total_produk = df['Nama Produk'].nunique()
    total_transaksi = df['No. Pesanan'].nunique()

    col1, col2 = st.columns(2)
    col1.metric("Jumlah Produk", total_produk)
    col2.metric("Jumlah Transaksi", total_transaksi)

    trx_count = df.groupby('No. Pesanan')['Nama Produk'].nunique()
    single_item = sum(trx_count == 1)
    multi_item = sum(trx_count > 1)

    col1, col2 = st.columns(2)
    col1.metric("Single-item Transactions", single_item)
    col2.metric("Multi-item Transactions", multi_item)

    st.write("Top produk terjual:")
    st.dataframe(df['Nama Produk'].value_counts().head(10))

    # cek kolom wajib
    if not all(col in df.columns for col in ['No. Pesanan', 'Nama Produk']):
        st.error("Dataset harus memuat kolom: `No. Pesanan` dan `Nama Produk`.")
        st.stop()

    # ----------------------------
    # PREPROCESSING
    # ----------------------------
    df = df[['No. Pesanan', 'Nama Produk']].dropna().drop_duplicates()
    df['Nama Produk'] = df['Nama Produk'].str.lower().str.strip()

    trx_list = df.groupby('No. Pesanan')['Nama Produk'].apply(list)
    trx_count = df.groupby('No. Pesanan')['Nama Produk'].nunique()

    single_list = trx_list[trx_count == 1].tolist()
    multi_list = trx_list[trx_count > 1].tolist()

    multi_balanced = resample(multi_list, replace=True, n_samples=len(single_list), random_state=42)
    transactions = single_list + multi_balanced

    # ----------------------------
    # TRANSACTION ENCODING
    # ----------------------------
    te = TransactionEncoder()
    encoded = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(encoded, columns=te.columns_).astype(int)

    # ----------------------------
    # PARAMETER INPUT
    # ----------------------------
    st.subheader("Parameter Apriori")
    col1, col2 = st.columns(2)
    min_support = col1.number_input("Minimum Support", 0.001, 1.0, 0.01, 0.01)
    min_confidence = col2.number_input("Minimum Confidence", 0.01, 1.0, 0.20, 0.05)

    # ----------------------------
    # RUN APRIORI
    # ----------------------------
    if st.button("Proses Apriori"):
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        st.write("Frequent Itemsets:")
        st.dataframe(frequent_itemsets)

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules_clean = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
        rules_clean['antecedents'] = rules_clean['antecedents'].apply(lambda x: ", ".join(list(x)))
        rules_clean['consequents'] = rules_clean['consequents'].apply(lambda x: ", ".join(list(x)))
        rules_clean = rules_clean.sort_values(by="confidence", ascending=False)

        st.write("Association Rules:")
        st.dataframe(rules_clean, hide_index=True)

        # ----------------------------
        # INTERPRETASI: TOP 3 RULES
        # ----------------------------
        st.subheader("Interpretasi Sederhana (Top 3)")

        if not rules_clean.empty:
            top3 = rules_clean.head(3)
            for _, row in top3.iterrows():
                antecedent = row['antecedents']
                consequent = row['consequents']
                conf = round(row['confidence'] * 100, 2)
                lift = round(row['lift'], 2)

                st.write(f"- Jika konsumen membeli **{antecedent}**, terdapat kemungkinan **{conf}%** konsumen juga membeli **{consequent}**.")
                st.write(f"  Nilai lift: {lift}")
                st.write("")

        else:
            st.write("Tidak ada aturan terbentuk. Silakan turunkan nilai support atau confidence.")

        # ----------------------------
        # DOWNLOAD
        # ----------------------------
        csv_rules = rules_clean.to_csv(index=False).encode('utf-8')
        st.download_button("Download Hasil Rules", csv_rules, "hasil_apriori.csv", "text/csv")
