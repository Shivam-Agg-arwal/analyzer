import re
import emoji
import pandas as pd
from datetime import datetime
from collections import Counter
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import plotly.express as px
import numpy as np

# -----------------------------
# Streamlit Config (Dark Mode)
# -----------------------------
st.set_page_config(page_title="📱 WhatsApp Chat Analyzer", layout="wide", initial_sidebar_state="expanded")

# Use dark background style
plt.style.use("dark_background")
sns.set_style("darkgrid")

# -----------------------------
# Chat Parsing
# -----------------------------
def parse_chat(text):
    pattern = r"(\d{2}/\d{2}/\d{4}), (\d{2}):(\d{2}) - ([^:]+): (.*)"
    matches = re.findall(pattern, text)

    chat_array = []
    for date_str, hour, minute, name, message in matches:
        dt = datetime.strptime(f"{date_str}, {hour}:{minute}", "%d/%m/%Y, %H:%M")
        total_minutes = int(dt.timestamp() // 60)
        chat_array.append({
            "Datetime": dt,
            "Date": dt.date(),
            "Minutes": total_minutes,
            "User": name.strip(),
            "Message": message.strip(),
            "Letters": len(message),
            "Words": len(message.split()),
            "Emoji_Count": sum(1 for ch in message if ch in emoji.EMOJI_DATA)
        })
    return pd.DataFrame(chat_array)

# -----------------------------
# Summaries
# -----------------------------
def summarize_data(df):
    total = {
        "Messages": len(df),
        "Words": df["Words"].sum(),
        "Letters": df["Letters"].sum(),
        "Emojis": df["Emoji_Count"].sum()
    }

    daywise = df.groupby(["Date", "User"]).agg({
        "Message": "count",
        "Words": "sum",
        "Letters": "sum",
        "Emoji_Count": "sum"
    }).reset_index().rename(columns={"Message": "Messages"})

    daywise["Emoji/msg %"] = (daywise["Emoji_Count"] / daywise["Messages"] * 100).round(2)
    daywise["Emoji/letters %"] = (daywise["Emoji_Count"] / daywise["Letters"] * 100).round(4)

    userwise = df.groupby("User").agg({
        "Message": "count",
        "Words": "sum",
        "Letters": "sum",
        "Emoji_Count": "sum"
    }).reset_index().rename(columns={"Message": "Messages"})

    userwise["Emoji/msg %"] = (userwise["Emoji_Count"] / userwise["Messages"] * 100).round(2)
    userwise["Emoji/letters %"] = (userwise["Emoji_Count"] / userwise["Letters"] * 100).round(4)

    return total, daywise, userwise

# -----------------------------
# Default Sample
# -----------------------------

default_chat = """20/09/2025, 10:00 - Shivam: Best of luck for exams! 🍀"""

# -----------------------------
# UI
# -----------------------------
st.title("🌙 WhatsApp Chat Analytics — Dark Mode")
st.markdown("Analyze chat patterns, emoji usage, word trends, and more — beautifully visualized in dark mode.")

uploaded = st.file_uploader("📂 Upload WhatsApp Chat (.txt)", type=["txt"])
if uploaded:
    text = uploaded.read().decode("utf-8")
else:
    text = default_chat
    st.info("📄 Using default demo chat")

df = parse_chat(text)
if df.empty:
    st.warning("No messages found!")
    st.stop()

total, daywise, userwise = summarize_data(df)

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "📊 Overview", "☁️ WordCloud", "⏰ Activity Heatmap",
    "😂 Emoji Stats", "⏱️ Reply Lag", "🧩 Timeline"
])

# -----------------------------
# Overview Tab
# -----------------------------
with tabs[0]:
    st.header("📈 Overview")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💬 Total Messages", total["Messages"])
    col2.metric("✍️ Total Words", total["Words"])
    col3.metric("🔠 Total Letters", total["Letters"])
    col4.metric("😂 Total Emojis", total["Emojis"])
    
    # Userwise totals
    st.subheader("👤 Userwise Totals")
    st.dataframe(userwise)
    
    # Messages per user
    fig = px.bar(userwise, x="User", y="Messages", color="User",
                 title="💬 Messages per User", color_discrete_sequence=px.colors.qualitative.Dark2)
    st.plotly_chart(fig, use_container_width=True)
    
    # Daily total messages
    st.subheader("📅 Total Messages per Day")
    daily_counts = df.groupby("Date")["Message"].count().reset_index().rename(columns={"Message": "Total_Messages"})
    fig_daily = px.bar(
        daily_counts,
        x="Date",
        y="Total_Messages",
        title="💬 Total Messages per Day",
        text="Total_Messages",
        color="Total_Messages",
        color_continuous_scale="Viridis"
    )
    fig_daily.update_layout(xaxis_title="Date", yaxis_title="Number of Messages", showlegend=False)
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # Daywise percentage contributions
    st.subheader("📊 Daywise User Contribution (%)")
    trend_df = daywise.copy()
    for metric in ["Messages", "Words", "Letters"]:
        trend_df[f"{metric}_%"] = trend_df.groupby("Date")[metric].transform(lambda x: x / x.sum() * 100)
    
    # Messages %
    fig1 = px.bar(
        trend_df,
        x="Date",
        y="Messages_%",
        color="User",
        title="💬 % of Messages per User by Day",
        text_auto=".1f",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig1.update_layout(barmode="stack", xaxis_title="Date", yaxis_title="% of Messages")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Words %
    fig2 = px.bar(
        trend_df,
        x="Date",
        y="Words_%",
        color="User",
        title="🗒️ % of Words per User by Day",
        text_auto=".1f",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig2.update_layout(barmode="stack", xaxis_title="Date", yaxis_title="% of Words")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Letters %
    fig3 = px.bar(
        trend_df,
        x="Date",
        y="Letters_%",
        color="User",
        title="🔠 % of Letters per User by Day",
        text_auto=".1f",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig3.update_layout(barmode="stack", xaxis_title="Date", yaxis_title="% of Letters")
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# WordCloud Tab
# -----------------------------
with tabs[1]:
    st.header("☁️ Word Cloud & Word Frequency")
    all_text = " ".join(df["Message"].tolist()).lower()
    stopwords = set(["the", "is", "and", "to", "a", "of", "in", "on", "it", "for", "that", "me", "i", "you", "my"])
    words = re.findall(r"\b[a-zA-Z']+\b", all_text)
    filtered = [w for w in words if w not in stopwords]
    freq = Counter(filtered).most_common(25)
    
    st.subheader("📋 Top Words")
    st.dataframe(pd.DataFrame(freq, columns=["Word", "Count"]))
    
    wc = WordCloud(width=800, height=400, background_color="black", colormap="plasma").generate(" ".join(filtered))
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# -----------------------------
# Activity Heatmap Tab
# -----------------------------
with tabs[2]:
    st.header("⏰ Hourly Message Heatmap")
    df["Hour"] = df["Datetime"].dt.hour
    pivot = df.pivot_table(index="User", columns="Hour", values="Message", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(pivot, cmap="rocket_r", ax=ax)
    ax.set_title("Messages per Hour (Dark Mode)")
    st.pyplot(fig)

# -----------------------------
# Emoji Stats Tab
# -----------------------------
with tabs[3]:
    st.header("😂 Emoji Breakdown")
    all_emojis = "".join(df["Message"].apply(lambda x: "".join(ch for ch in x if ch in emoji.EMOJI_DATA)))
    counts = Counter(all_emojis)
    top10 = counts.most_common(10)
    if top10:
        emoji_df = pd.DataFrame(top10, columns=["Emoji", "Count"])
        fig = px.pie(emoji_df, values="Count", names="Emoji", title="Top Emojis",
                     color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig)
        st.dataframe(emoji_df)
    else:
        st.info("No emojis found in this chat.")

# -----------------------------
# Reply Lag Tab
# -----------------------------
with tabs[4]:
    st.header("⏱️ Response Lag (Avg Reply Time)")
    lags = []
    for i in range(1, len(df)):
        prev_user = df.iloc[i-1]["User"]
        curr_user = df.iloc[i]["User"]
        if prev_user != curr_user:
            lags.append([prev_user, curr_user, df.iloc[i]["Minutes"] - df.iloc[i-1]["Minutes"]])
    if lags:
        lag_df = pd.DataFrame(lags, columns=["Sender", "Responder", "Lag_Min"])
        avg_lag = lag_df.groupby("Responder")["Lag_Min"].mean().round(2)
        st.dataframe(avg_lag)
        fig = px.bar(avg_lag, x=avg_lag.index, y="Lag_Min",
                     title="Average Reply Time (Minutes)", color=avg_lag.index)
        st.plotly_chart(fig)
    else:
        st.info("Not enough alternating messages to calculate reply lag.")

# -----------------------------
# Timeline Tab
# -----------------------------
with tabs[5]:
    st.header("🧩 Conversation Timeline")
    df["Timestamp"] = df["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
    fig = px.scatter(df, x="Timestamp", y="User", color="User",
                     hover_data=["Message"], title="Chat Timeline",
                     color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig, use_container_width=True)
