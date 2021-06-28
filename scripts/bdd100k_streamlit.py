import json
import streamlit as st
import plotly.express as px
import plotly.io as pio
import pandas as pd
from pandas import json_normalize
import altair as alt



json_open = open('/hiro/bdd100k/bdd100k_det_20_labels_trainval/bdd100k/labels/det_20/det_val.json', 'r')
json_load = json.load(json_open)
print(json_load[0]['name'])
print()
labels = json_load[0]['labels']
print(labels)
print(len(labels))

# data
@st.cache
def load_data():
    df = json_normalize(labels)
    return df

df = load_data()
print(df.head())

st.title('BDD100K labels')

#st.write("sample code")
# sidemenu
st.sidebar.markdown(
    "# Sidebar"
)

template = st.sidebar.selectbox(
    "Template", list(pio.templates.keys())
)

category = st.sidebar.selectbox(
    "Category", list(df.category.unique())
)

filtered_df = df[df.category == category]
st.write(
    px.bar(filtered_df, x="box2d.x1", y="box2d.y1", color="attributes.occluded", barmode="group", template=template)
)

category = list(df.category.unique())
selected_category  = st.multiselect('select targets', category , default=category )
df = df[df.category.isin(selected_category)]

st.dataframe(df)
# python bdd100k_streamlit.py

