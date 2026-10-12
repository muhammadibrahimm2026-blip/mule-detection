import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Page Layout Configuration
st.set_page_config(page_title="Mule Account Network Detector", layout="wide")

st.title("🚨 Mule Account Network Detector")
st.markdown("### AI-Powered Graph Analytics for Anti-Money Laundering (AML)")

# Sidebar Controls for Hackathon Interactivity
st.sidebar.header("Simulation Settings")
num_accounts = st.sidebar.slider("Number of Accounts", 100, 500, 350, 50)
contamination_rate = st.sidebar.slider("Anomaly Contamination Rate", 0.01, 0.10, 0.03, 0.01)

# Main Execution Function (Cached for performance)
@st.cache_data
def run_detection(n_acc, contam):
    np.random.seed(42)
    accounts = [f"ACC_{i:03d}" for i in range(n_acc)]
    transactions = []

    # 1. Background normal activity
    for _ in range(n_acc * 4):
        sender, receiver = np.random.choice(accounts, 2, replace=False)
        transactions.append({
            'sender': sender, 'receiver': receiver,
            'amount': np.random.exponential(scale=120) + 5,
            'timestamp': np.random.randint(0, 86400), 'is_mule': 0
        })

    # 2. Injecting a Mule Ring
    mule_ring = ['ACC_012', 'ACC_045', 'ACC_089', 'ACC_154', 'ACC_210', 'ACC_288']
    for i in range(len(mule_ring) - 1):
        for _ in range(12):
            transactions.append({
                'sender': mule_ring[i], 'receiver': mule_ring[i+1],
                'amount': np.random.randint(8000, 9999),
                'timestamp': np.random.randint(3000, 4500), 'is_mule': 1
            })

    df_txn = pd.DataFrame(transactions)

    # 3. Graph Build & Feature Extraction
    G = nx.DiGraph()
    for _, row in df_txn.iterrows():
        if G.has_edge(row['sender'], row['receiver']):
            G[row['sender']][row['receiver']]['weight'] += row['amount']
            G[row['sender']][row['receiver']]['count'] += 1
        else:
            G.add_edge(row['sender'], row['receiver'], weight=row['amount'], count=1)

    try:
        hubs, authorities = nx.hits(G, max_iter=2000, tol=1e-06)
    except:
        hubs, authorities = {}, {}

    feature_data = []
    for node in G.nodes():
        feature_data.append({
            'account': node,
            'in_degree': G.in_degree(node),
            'out_degree': G.out_degree(node),
            'pagerank': nx.pagerank(G, alpha=0.85).get(node, 0),
            'hub_score': hubs.get(node, 0.0),
            'auth_score': authorities.get(node, 0.0),
            'is_true_mule': 1 if node in mule_ring else 0
        })

    df_features = pd.DataFrame(feature_data)
    
    # 4. Unsupervised Detection Model
    X = df_features[['in_degree', 'out_degree', 'pagerank', 'hub_score', 'auth_score']]
    iso_forest = IsolationForest(contamination=contam, random_state=42)
    df_features['anomaly_score'] = iso_forest.fit_predict(X)
    df_features['is_flagged'] = df_features['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

    return df_features, mule_ring

df_features, mule_ring = run_detection(num_accounts, contamination_rate)
flagged_count = int(df_features['is_flagged'].sum())

# Top Metrics Bar
col1, col2, col3 = st.columns(3)
col1.metric("Total Accounts Analyzed", len(df_features))
col2.metric("High-Risk Accounts Flagged", flagged_count)
col3.metric("Injected Mule Ring Size", len(mule_ring))

st.markdown("---")

# Render Analytical Dashboard (Your Step 4 Visuals)
st.subheader("📊 Graph Topology & Detection Dashboard")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left Plot: Degree Scatter
colors = ['crimson' if flag == 1 else 'dodgerblue' for flag in df_features['is_flagged']]
axes[0].scatter(df_features['in_degree'], df_features['out_degree'], c=colors, alpha=0.7, edgecolors='k', s=50)
axes[0].set_title("Degree Topology Scatter (Flagged Anomalies in Red)", fontweight='bold')
axes[0].set_xlabel("In-Degree (Money Received)")
axes[0].set_ylabel("Out-Degree (Money Sent)")

# Right Plot: PageRank Ranking Barplot
top_hubs = df_features.sort_values(by='pagerank', ascending=False).head(10)
axes[1].barh(top_hubs['account'], top_hubs['pagerank'], color='mediumpurple')
axes[1].set_title("Top 10 Central Accounts by PageRank Score", fontweight='bold')
axes[1].set_xlabel("PageRank Centrality Score")
axes[1].invert_yaxis()

plt.tight_layout()
st.pyplot(fig)