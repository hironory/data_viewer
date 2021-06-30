import json
import streamlit as st
import plotly.express as px
import plotly.io as pio
import pandas as pd
from pandas import json_normalize
import altair as alt
import matplotlib.pyplot as plt


json_open = open('/hiro/bdd100k/bdd100k_det_20_labels_trainval/bdd100k/labels/det_20/det_val.json', 'r')
json_load = json.load(json_open)

print("file num =", len(json_load))
print(json_load[0]['name'])
print()
labels = json_load[0]['labels']
#labels[0]["file_name"] = json_load[0]['name']
#print(labels)
print()

DATA_NUM = len(json_load)
DATA_NUM = 50

all_labels_json = []
for i in range(DATA_NUM):
    #add file name
    for lab in json_load[i]['labels']:
        lab["file_name"] = json_load[i]['name']

    all_labels_json += json_load[i]['labels']

# data
@st.cache
def load_data(input):
    df = json_normalize(input)
    df['box_size'] = (df['box2d.x2'] - df['box2d.x1'])*(df['box2d.y2'] - df['box2d.y1'])
    return df

df = load_data(all_labels_json)



print("number of objects =", len(df))


print(df.head())

st.title('BDD100K Object Detection labels')

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


st.write(' ## Number of objects per category') # markdown
category_df = df['category'].value_counts().to_frame().reset_index()
category_df = category_df.rename(columns={'index':'category', 'category':'num of objects'})
st.write(
    px.bar(category_df, x="category", y="num of objects", template=template)
    )



st.write(' ## object size histogram of', category) # markdown
filtered_df = df[df.category == category]
st.write(
    px.histogram(filtered_df, x="box_size", template=template)
    )



#st.write(
#    px.bar(filtered_df, x="box2d.x1", y="box2d.y1", color="attributes.occluded", barmode="group", template=template)
#)



st.write(' ## data frame') # markdown
category = list(df.category.unique())
selected_category  = st.multiselect('select targets', category , default=category )
df = df[df.category.isin(selected_category)]
st.dataframe(df)
# python bdd100k_streamlit.py

