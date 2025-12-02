import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils import resample
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------------
# TITLE & DESCRIPTION
# -----------------------------------
st.title("📊 Analisis Association Rule dengan Apriori")
st.write("Upload dataset transaksi dan jalankan metode Apriori untuk menemukan pola pembelian konsumen.")

# -----------------------------------
# UPLOAD FILE
# -----------------------------------
uploaded = st.file_uploader("Upload File Dataset (.xlsx / .csv)", type=["xlsx", "csv"])

if uploaded:
    st.success("File berhasil diupload!")

    # -----------------------------------
    # LOAD DATAFRAME
    # -----------------------------------
    if uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    st.subheader("📁 Data Awal")
    st.dataframe(df.head())

    st.subheader("Produk Terlaris")
    top_products = df['Nama Produk'].value_counts().head(10)
    st.dataframe(top_products)

    total_transaksi = df['No. Pesanan'].nunique()
    total_produk = df['Nama Produk'].nunique()
    st.write(f"Jumlah transaksi: {total_transaksi}")
    st.write(f"Jumlah produk : {total_produk}")

    trx_count = df.groupby('No. Pesanan')['Nama Produk'].nunique()
    single_item = sum(trx_count == 1)
    multi_item  = sum(trx_count > 1)
    st.write(f"Single-item: {single_item}")
    st.write(f"Multi-item : {multi_item}")

    # -----------------------------------
    # SELECTION
    # -----------------------------------
    required_cols = ['No. Pesanan', 'Nama Produk']

    if not all(col in df.columns for col in required_cols):
        st.error(f"Kolom wajib tidak ditemukan: {required_cols}")
        st.stop()

    # -----------------------------------
    # CLEANING
    # -----------------------------------
    df = df[['No. Pesanan', 'Nama Produk']].dropna().drop_duplicates()
    df['Nama Produk'] = df['Nama Produk'].str.lower().str.strip()

    # -----------------------------------
    # OVERSAMPLING TRANSAKSI
    # -----------------------------------
    st.subheader("🔧 Oversampling (Menyamakan jumlah transaksi single-item & multi-item)")

    trx_count = df.groupby('No. Pesanan')['Nama Produk'].nunique()
    trx_list = df.groupby('No. Pesanan')['Nama Produk'].apply(list)

    single_list = trx_list[trx_count == 1].tolist()
    multi_list = trx_list[trx_count > 1].tolist()

    target_count = len(single_list)

    # Oversampling multi-transaction
    multi_oversampled = resample(multi_list, replace=True, n_samples=target_count, random_state=42)

    # Gabungkan transaksi seimbang
    balanced_transactions = single_list + multi_oversampled

    st.write(f"Single-item: {len(single_list)} transaksi")
    st.write(f"Multi-item : {len(multi_oversampled)} transaksi")

    # -----------------------------------
    # TRANSACTION ENCODING
    # -----------------------------------
    te = TransactionEncoder()
    te_ary = te.fit(balanced_transactions).transform(balanced_transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_).astype(int)

    st.subheader("📌 Data Setelah Encoding")
    st.dataframe(df_encoded.head())

    # -----------------------------------
    # INPUT PARAMETER APRIORI
    # -----------------------------------
    st.subheader("⚙️ Parameter Apriori")

    min_support = st.number_input("Minimum Support", min_value=0.001, max_value=1.0, value=0.01, step=0.01)
    min_confidence = st.number_input("Minimum Confidence", min_value=0.01, max_value=1.0, value=0.20, step=0.05)

    if st.button("🔍 Jalankan Apriori"):
        # -----------------------------------
        # APRIORI
        # -----------------------------------
        st.subheader("📌 Frequent Itemsets")

        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

        st.dataframe(frequent_itemsets)

        # -----------------------------------
        # RULES
        # -----------------------------------
        st.subheader("📈 Association Rules")

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        # Format tabel
        rules_clean = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
        rules_clean['antecedents'] = rules_clean['antecedents'].apply(lambda x: ", ".join(list(x)))
        rules_clean['consequents'] = rules_clean['consequents'].apply(lambda x: ", ".join(list(x)))
        rules_clean = rules_clean.sort_values(by="confidence", ascending=False)

        st.dataframe(rules_clean)

        # -----------------------------------
        # DOWNLOADABLE RESULT
        # -----------------------------------
        csv_rules = rules_clean.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="⬇️ Download Hasil Rules (CSV)",
            data=csv_rules,
            file_name="apriori_rules.csv",
            mime="text/csv"
        )

