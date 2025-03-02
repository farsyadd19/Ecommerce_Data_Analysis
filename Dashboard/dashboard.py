# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from babel.numbers import format_currency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Judul Dashboard
st.set_page_config(page_title="E-Commerce Data Analysis Dashboard", layout="wide")
st.title("📊 E-Commerce Data Analysis Dashboard")
st.markdown("""
    Selamat datang di dashboard analisis data e-commerce!  
    Dashboard ini menyediakan wawasan mendalam tentang tren penjualan, kategori produk, rating pelanggan, segmentasi pelanggan, dan distribusi geografis.
""")

# Sidebar untuk Personalisasi
with st.sidebar:
    st.title("👤 Farhan Rasyad")
    st.image("Dashboard/e-commerce.jpeg", caption="E-Commerce Dashboard", use_container_width=True)
    
    # Input Nama Pengguna
    user_name = st.text_input("Masukkan Nama Anda", "Pengguna")
    st.write(f"Halo, {user_name}! Selamat datang di dashboard kami.")

    # Load dataset
    all_data = pd.read_csv("Dashboard/all_data.csv")
    customer_location_data = pd.read_csv("Data/Geolocation.csv")

    datetime_cols = [
        "order_approved_at", 
        "order_delivered_carrier_date", 
        "order_delivered_customer_date", 
        "order_estimated_delivery_date", 
        "order_purchase_timestamp", 
        "shipping_limit_date"
    ]
    for col in datetime_cols:
        all_data[col] = pd.to_datetime(all_data[col], errors='coerce')  # Handle invalid dates

    min_date = all_data["order_approved_at"].min().date()
    max_date = all_data["order_approved_at"].max().date()

    # Date Range Filter
    start_date, end_date = st.date_input(
        label="Pilih Rentang Tanggal",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
     # Mode Gelap/Terang
    dark_mode = st.checkbox("Mode Gelap")
    if dark_mode:
        st.markdown("""
            <style>
            body {
                background-color: #121212;
                color: #ffffff;
            }
            .stApp {
                background-color: #121212;
            }
            .css-1d391kg {
                background-color: #1e1e1e;
            }
            </style>
            """, unsafe_allow_html=True)

# Konversi start_date dan end_date ke Timestamp
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

# Filter Data Berdasarkan Tanggal
filtered_df = all_data[
    (all_data["order_approved_at"] >= start_date) & 
    (all_data["order_approved_at"] <= end_date)
]

# Overview E-Commerce: KPI Metrics
st.header("Overview E-Commerce")

# Hitung nilai-nilai KPI
total_buyers = filtered_df['customer_unique_id'].nunique()
avg_review_score = filtered_df['review_score'].mean() if 'review_score' in filtered_df.columns else 0
total_sales_revenue = filtered_df['payment_value'].sum()
total_orders = filtered_df['order_id'].nunique()
total_product_categories = filtered_df['product_category_name_english'].nunique()

# Tampilkan KPI dalam kolom
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(label="Total Buyers", value=total_buyers)

with col2:
    st.metric(label="Avg Review Score", value=f"{avg_review_score:.2f}")

with col3:
    st.metric(label="Total Sales Revenue", value=format_currency(total_sales_revenue, 'BRL', locale='pt_BR'))

with col4:
    st.metric(label="Total Orders", value=total_orders)

with col5:
    st.metric(label="Total Product Categories", value=total_product_categories)

# Visualisasi 1: Tren Pertumbuhan Penjualan
st.header("1. Tren Pertumbuhan Penjualan")
monthly_revenue = filtered_df.resample(rule='M', on='order_approved_at').agg({"payment_value": "sum"}).reset_index()
monthly_revenue['month_year'] = monthly_revenue['order_approved_at'].dt.strftime('%B %Y')

plt.figure(figsize=(14, 7))
sns.lineplot(
    x=monthly_revenue['month_year'],
    y=monthly_revenue['payment_value'],
    marker='o',
    linewidth=2.5,
    color="royalblue",
    label="Total Pendapatan"
)

for x, y in zip(monthly_revenue['month_year'], monthly_revenue['payment_value']):
    plt.scatter(x, y, color='gold', s=80, edgecolors='black', zorder=3)

for i, txt in enumerate(monthly_revenue['payment_value']):
    plt.annotate(f"{txt:,.0f}",
                 (monthly_revenue['month_year'][i], monthly_revenue['payment_value'][i]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center',
                 fontsize=10,
                 color="darkred")

plt.title("Tren Pertumbuhan Total Pendapatan E-Commerce", fontsize=18, fontweight='bold', color='darkblue')
plt.xlabel("Bulan", fontsize=14, fontweight='bold')
plt.ylabel("Total Pendapatan", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(plt)

# Visualisasi 2: Distribusi Penjualan Berdasarkan Kategori Produk
st.header("2. Distribusi Penjualan Berdasarkan Kategori Produk")
product_sales_summary = filtered_df.groupby("product_category_name_english")["product_id"].count().reset_index()
product_sales_summary.rename(columns={"product_id": "total_products_sold"}, inplace=True)
product_sales_summary = product_sales_summary.sort_values(by="total_products_sold", ascending=False)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

sns.barplot(data=product_sales_summary.head(5), x="total_products_sold", y="product_category_name_english", palette="Blues_r", ax=axes[0])
axes[0].set_title("Top 5 Kategori Produk Terlaris", fontsize=18)
axes[0].set_xlabel("Jumlah Produk Terjual", fontsize=14)
axes[0].set_ylabel(None)

sns.barplot(data=product_sales_summary.tail(5), x="total_products_sold", y="product_category_name_english", palette="Oranges_r", ax=axes[1])
axes[1].set_title("Bottom 5 Kategori Produk Kurang Laku", fontsize=18)
axes[1].set_xlabel("Jumlah Produk Terjual", fontsize=14)
axes[1].set_ylabel(None)
axes[1].invert_xaxis()
axes[1].yaxis.tick_right()

plt.suptitle("Distribusi Penjualan Berdasarkan Kategori Produk", fontsize=25)
plt.tight_layout(rect=[0, 0, 1, 0.96])
st.pyplot(fig)

# Visualisasi 3: Distribusi Rating Pelanggan
st.header("3. Distribusi Rating Pelanggan")
if 'review_score' in filtered_df.columns:
    rating_distribution = filtered_df['review_score'].value_counts().sort_index().reset_index()
    rating_distribution.columns = ['rating', 'count']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=rating_distribution, x='rating', y='count', palette="viridis", ax=ax)
    ax.set_title('Distribusi Rating Pelanggan', fontsize=18, fontweight='bold')
    ax.set_xlabel('Rating', fontsize=14)
    ax.set_ylabel('Jumlah Pesanan', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    # Rata-rata rating per kategori produk
    average_rating_per_category = filtered_df.groupby("product_category_name_english")["review_score"].mean().reset_index()
    average_rating_per_category = average_rating_per_category.sort_values(by="review_score", ascending=False)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

    sns.barplot(data=average_rating_per_category.head(5), x="review_score", y="product_category_name_english", palette="Greens_r", ax=axes[0])
    axes[0].set_title("Top 5 Kategori dengan Rating Terbaik", fontsize=18)
    axes[0].set_xlabel("Rata-rata Rating", fontsize=14)
    axes[0].set_ylabel(None)

    sns.barplot(data=average_rating_per_category.tail(5), x="review_score", y="product_category_name_english", palette="Reds_r", ax=axes[1])
    axes[1].set_title("Bottom 5 Kategori dengan Rating Terburuk", fontsize=18)
    axes[1].set_xlabel("Rata-rata Rating", fontsize=14)
    axes[1].set_ylabel(None)
    axes[1].invert_xaxis()
    axes[1].yaxis.tick_right()

    plt.suptitle("Rating Pelanggan Berdasarkan Kategori Produk", fontsize=25)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)
else:
    st.write("Kolom 'review_score' tidak ditemukan dalam dataset.")

# Visualisasi 4: Segmentasi Pelanggan (RFM)
st.header("4. Segmentasi Pelanggan (RFM)")
tanggal_acuan = filtered_df['order_approved_at'].max() + pd.Timedelta(days=1)
data_rfm = (
    filtered_df.groupby('customer_unique_id')
    .agg(
        waktu_terakhir=('order_approved_at', lambda x: (tanggal_acuan - x.max()).days),
        total_transaksi=('order_id', 'nunique'),
        total_pengeluaran=('price', 'sum')
    )
    .reset_index()
)

data_rfm['skor_waktu'] = pd.qcut(data_rfm['waktu_terakhir'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
data_rfm['skor_transaksi'] = pd.qcut(data_rfm['total_transaksi'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
data_rfm['skor_pengeluaran'] = pd.qcut(data_rfm['total_pengeluaran'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
data_rfm['total_skor'] = data_rfm['skor_waktu'] + data_rfm['skor_transaksi'] + data_rfm['skor_pengeluaran']

def klasifikasi_pelanggan(score):
    if score >= 12:
        return 'Pelanggan Premium'
    elif 9 <= score < 12:
        return 'Pelanggan Loyal'
    elif 7 <= score < 9:
        return 'Pelanggan Potensial'
    elif 5 <= score < 7:
        return 'Pelanggan Rawan'
    else:
        return 'Pelanggan Tidak Aktif'

data_rfm['klasifikasi'] = data_rfm['total_skor'].apply(klasifikasi_pelanggan)
distribusi_segmen = data_rfm['klasifikasi'].value_counts().reindex(
    ['Pelanggan Premium', 'Pelanggan Loyal', 'Pelanggan Potensial', 'Pelanggan Rawan', 'Pelanggan Tidak Aktif']
)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    x=distribusi_segmen.index,
    y=distribusi_segmen.values,
    palette=["#2ECC71", "#1ABC9C", "#F1C40F", "#E67E22", "#E74C3C"]
)
ax.set_title("Distribusi Segmen Pelanggan Berdasarkan Analisis RFM", fontsize=18, fontweight='bold')
ax.set_xlabel("Kategori Pelanggan", fontsize=14)
ax.set_ylabel("Jumlah Pelanggan", fontsize=14)
ax.tick_params(axis='x', rotation=20)
ax.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig)

# Visualisasi 5: Distribusi Pelanggan Berdasarkan Negara Bagian
st.header("5. Distribusi Pelanggan Berdasarkan Negara Bagian")
distribusi_pelanggan_negara_bagian = (
    filtered_df.groupby('customer_state')
    .agg(total_pelanggan=('customer_unique_id', 'nunique'), total_pembelian=('payment_value', 'sum'))
    .reset_index()
    .sort_values(by='total_pembelian', ascending=False)
)

plt.figure(figsize=(14, 8))
sns.barplot(
    data=distribusi_pelanggan_negara_bagian,
    x='customer_state',
    y='total_pembelian',
    palette="viridis"
)
plt.title("Distribusi Pelanggan Berdasarkan Negara Bagian di Brasil", fontsize=18, fontweight='bold')
plt.xlabel("Negara Bagian", fontsize=14)
plt.ylabel("Total Pembelian", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
st.pyplot(plt)

# Visualisasi 6: Top 10 Kota dengan Total Pembelian Tertinggi
st.header("6. Top 10 Kota dengan Total Pembelian Tertinggi")
distribusi_pelanggan_kota = (
    filtered_df.groupby('customer_city')
    .agg(total_pelanggan=('customer_unique_id', 'nunique'), total_pembelian=('payment_value', 'sum'))
    .reset_index()
    .sort_values(by='total_pembelian', ascending=False)
    .head(10)
)

plt.figure(figsize=(14, 8))
sns.barplot(
    data=distribusi_pelanggan_kota,
    x='customer_city',
    y='total_pembelian',
    palette="coolwarm"
)
plt.title("Top 10 Kota dengan Total Pembelian Tertinggi di Brasil", fontsize=18, fontweight='bold')
plt.xlabel("Kota", fontsize=14)
plt.ylabel("Total Pembelian", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
st.pyplot(plt)

# Visualisasi 7: Pola Pembelian Berdasarkan Wilayah
st.header("7. Pola Pembelian Berdasarkan Wilayah")
data_rfm = (
    filtered_df.groupby('customer_unique_id')
    .agg(
        waktu_terakhir=('order_purchase_timestamp', lambda x: (pd.to_datetime('today') - pd.to_datetime(x.max())).days),
        total_transaksi=('order_id', 'nunique'),
        total_pengeluaran=('payment_value', 'sum')
    )
    .reset_index()
)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_rfm[['waktu_terakhir', 'total_transaksi', 'total_pengeluaran']].dropna())

kmeans = KMeans(n_clusters=5, random_state=42)
data_rfm['cluster'] = kmeans.fit_predict(data_scaled)

def label_cluster(cluster):
    if cluster == 0:
        return 'Cluster Tinggi'
    elif cluster == 1:
        return 'Cluster Sedang-Tinggi'
    elif cluster == 2:
        return 'Cluster Sedang-Rendah'
    elif cluster == 3:
        return 'Cluster Rendah'
    else:
        return 'Cluster Sangat Rendah'

data_rfm['label_cluster'] = data_rfm['cluster'].apply(label_cluster)

merged_rfm_location = pd.merge(
    left=data_rfm,
    right=filtered_df[['customer_unique_id', 'customer_city', 'customer_state']].drop_duplicates(),
    how="left",
    on="customer_unique_id"
)

pola_pembelian_wilayah = (
    merged_rfm_location.groupby(['customer_state', 'label_cluster'])
    .agg(total_pembelian=('total_pengeluaran', 'sum'), jumlah_pelanggan=('customer_unique_id', 'nunique'))
    .reset_index()
)

plt.figure(figsize=(14, 8))
sns.barplot(
    data=pola_pembelian_wilayah,
    x='customer_state',
    y='total_pembelian',
    hue='label_cluster',
    palette="viridis"
)
plt.title("Pola Pembelian Berdasarkan Wilayah di Brasil", fontsize=18, fontweight='bold')
plt.xlabel("Negara Bagian", fontsize=14)
plt.ylabel("Total Pembelian", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.legend(title="Cluster Pembelian", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
st.pyplot(plt)



# Visualisasi 8: Heatmap Persebaran Pelanggan
st.header("8. Heatmap Persebaran Pelanggan")
m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)
heat_data = [[row['geolocation_lat'], row['geolocation_lng']] for index, row in customer_location_data.iterrows()]
HeatMap(heat_data).add_to(m)
folium_static(m)


# Fitur Download Data
if st.sidebar.button("Unduh Data Terfilter"):
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Klik untuk Unduh",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )

# Feedback dari Pengguna
with st.sidebar.form("Feedback"):
    feedback = st.text_area("Masukkan Feedback Anda")
    submit_feedback = st.form_submit_button("Kirim Feedback")
    if submit_feedback:
        st.success("Terima kasih atas feedback Anda!")


# Footer
st.markdown("---")
st.markdown("Dashboard ini dibuat oleh **Farhan Rasyad** menggunakan **Streamlit** dan **Matplotlib/Seaborn** untuk visualisasi.")

st.caption('Copyright (C) Farhan Rasyad 2025')