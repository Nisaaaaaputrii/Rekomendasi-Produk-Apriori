import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils import resample
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# tampilan app
st.set_page_config(page_title="Analisis Apriori", layout="wide")

st.set_page_config(page_title="Analisis Apriori", layout="wide")

# tampilan UI
st.markdown("""
<style>

    /* background utama aplikasi */
    .stApp {
        background-color: #FFF1D6 !important;
    }

    /* container konten (agar putih dan rapi) */
    .main > div {
        background-color: white !important;
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0px 0px 8px rgba(0,0,0,0.05);
    }

    /* style button */
    .stButton>button {
        background-color: #FF8C00;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 18px;
        font-size: 15px;
    }
    .stButton>button:hover {
        background-color: #e67800;
        color: white;
    }

    /* style header text (judul tetap minimalis) */
    h1, h2, h3, h4 {
        color: #C46500;
    }

</style>
""", unsafe_allow_html=True)

# proses

st.title("Sistem Rekomendasi Produk - Apriori")
st.write("Unggah dataset transaksi untuk menemukan pola pembelian konsumen.")
uploaded = st.file_uploader("Pilih File Dataset (.xlsx / .csv)", type=["xlsx", "csv"])

if uploaded:

    if uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    colA, colB = st.columns(2)
    colA.metric("Jumlah Produk", df['Nama Produk'].nunique())
    colB.metric("Jumlah Transaksi", df['No. Pesanan'].nunique())

    trx_count = df.groupby('No. Pesanan')['Nama Produk'].nunique()
    single_item = sum(trx_count == 1)
    multi_item = sum(trx_count > 1)

    col1, col2 = st.columns(2)
    col1.metric("Transaksi Single-item", single_item)
    col2.metric("Transaksi Multi-item", multi_item)

    st.subheader("Top Produk Terlaris")
    st.dataframe(df['Nama Produk'].value_counts().head(10))


    if not all(col in df.columns for col in ['No. Pesanan', 'Nama Produk']):
        st.error("Dataset harus memiliki kolom: `No. Pesanan` dan `Nama Produk`.")
        st.stop()


    df = df[['No. Pesanan', 'Nama Produk']].dropna().drop_duplicates()
    df['Nama Produk'] = df['Nama Produk'].str.lower().str.strip()

    trx_list = df.groupby('No. Pesanan')['Nama Produk'].apply(list)
    trx_count = df.groupby('No. Pesanan')['Nama Produk'].nunique()

    single_list = trx_list[trx_count == 1].tolist()
    multi_list = trx_list[trx_count > 1].tolist()

    multi_balanced = resample(multi_list, replace=True, n_samples=len(single_list), random_state=42)
    transactions = single_list + multi_balanced


    te = TransactionEncoder()
    encoded = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(encoded, columns=te.columns_).astype(int)


    st.subheader("Parameter Apriori")
    colS, colC = st.columns(2)
    min_support = colS.number_input("Minimum Support", 0.001, 1.0, 0.01, 0.01)
    min_confidence = colC.number_input("Minimum Confidence", 0.01, 1.0, 0.20, 0.05)


    if st.button("🔍 Proses Apriori"):

        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)


        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules_clean = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()

        rules_clean['antecedents'] = rules_clean['antecedents'].apply(lambda x: ", ".join(list(x)))
        rules_clean['consequents'] = rules_clean['consequents'].apply(lambda x: ", ".join(list(x)))
        rules_clean = rules_clean.sort_values(by="confidence", ascending=False)

        st.write("Association Rules:")
        st.dataframe(rules_clean, hide_index=True)

        st.subheader("Interpretasi (Top 3)")

        if not rules_clean.empty:
            top3 = rules_clean.head(3)

            for _, row in top3.iterrows():
                antecedent = row['antecedents']
                consequent = row['consequents']
                conf = round(row['confidence'] * 100, 2)
                lift = round(row['lift'], 2)

                st.write(f"- Jika konsumen membeli **{antecedent}**, maka sekitar **{conf}%** kemungkinan juga membeli **{consequent}**.")
                

                if lift > 1:
                    st.write(f"  Nilai lift: {lift} → Hubungan pembelian **kuat**, bukan kebetulan. Produk cocok dijadikan bundling/paket.")
                elif lift == 1:
                    st.write(f"  Nilai lift: {lift} → Hubungan pembelian **netral**, pembelian wajar tanpa saling mempengaruhi.")
                else:
                    st.write(f"  Nilai lift: {lift} → Hubungan pembelian **lemah**, pola tidak signifikan sehingga tidak cocok dijadikan rekomendasi.")
                st.write("")

        else:
            st.write("Tidak ada aturan terbentuk. Silakan turunkan nilai support atau confidence.")

        
        csv_rules = rules_clean.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Hasil Rules", csv_rules, "hasil_apriori.csv", "text/csv")