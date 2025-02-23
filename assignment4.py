import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Titanic Data Dashboard")

df = pd.read_csv("/Users/veronikasippala/Desktop/TitanicData2.csv", encoding="ISO-8859-1", sep=";", on_bad_lines="skip")

with st.sidebar:
    st.header("Dataset Statistics")
    total_passengers = len(df)
    survivors = df["Survived"].sum()
    survival_rate = survivors / total_passengers * 100

    st.metric(label="Total Passengers", value=total_passengers)
    st.metric(label="Survivors", value=survivors)
    st.metric(label="Survival Rate (%)", value=f"{survival_rate:.2f}%")

st.write("## Titanic Data Preview")
st.dataframe(df.head())

st.write("## Survival Distribution")
fig, ax = plt.subplots()
df["Survived"].value_counts().plot(kind="bar", color=["red", "green"], ax=ax)
ax.set_xticklabels(["Did not survive", "Survived"], rotation=0)
ax.set_ylabel("Number of passengers")
st.pyplot(fig)

st.write("## Gender Distribution")
fig, ax = plt.subplots()
df["Sex"].value_counts().plot(kind="bar", ax=ax, color=["blue", "pink"])
ax.set_ylabel("Number of Passengers")
st.pyplot(fig)

pclass = st.selectbox("Select Passenger Class", df["Pclass"].unique())
filtered_df = df[df["Pclass"] == pclass]

st.write(f"### Passengers in Class {pclass}")
st.dataframe(filtered_df)

st.write("## Survival Rate by Passenger Class")
fig, ax = plt.subplots()
df.groupby("Pclass")["Survived"].mean().plot(kind="bar", ax=ax, color="purple")
ax.set_ylabel("Survival Rate")
st.pyplot(fig)

column = st.selectbox(" Select Column for Histogram", df.select_dtypes(include=["number"]).columns)

fig, ax = plt.subplots()
df[column].hist(ax=ax, bins=20)
ax.set_xlabel(column)
ax.set_ylabel("Frequency")
st.pyplot(fig)

if st.checkbox("Show raw dataset"):
    st.write("## Full Titanic Dataset")
    st.dataframe(df)
