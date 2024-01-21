import numpy as np
import pandas as pd
import streamlit as st
import st_pages as stp
import matplotlib.pyplot as plt
import seaborn as sns

sidebar_list=("Dashboard",
              "Data analysys",
              "Data cleanup",
              "Data visualisation",
              "Regresion model")
stp.add_page_title()
stp.show_pages([
    stp.Page("my_app\dashboard.py", sidebar_list[0],icon= ":notebook:"),
    stp.Section(name="Data preparation", icon=":books:"),
    stp.Page("my_app\pages\data_analyse.py", sidebar_list[1], icon=":books:",in_section=True),
    stp.Page("my_app\pages\data_cleanup.py", sidebar_list[2],icon= ":books:",in_section=True),
    stp.Page("my_app\pages\data_visualization.py", sidebar_list[3], icon=":bar_chart:",in_section=True),
    stp.Section(name="Regression model", icon=":bar_chart:"),
    stp.Page("my_app\pages\\regresion_model.py", sidebar_list[4],icon= ":books:",in_section=True)
])
df = pd.read_csv("data\messy_data.csv")
df.columns=df.columns.str.replace(' ','')
df.rename(columns={
    'xdimension':'x_dimension',
    'ydimension':'y_dimension',
    'zdimension':'z_dimension',
},inplace=True)
df=df.replace(r'^\s*$', np.nan, regex=True)
df['clarity']=df['clarity'].astype('category')
df['color']=df['color'].astype('category')
df['cut']=df['cut'].astype('category')
df['x_dimension']=df['x_dimension'].astype('float64')
df['y_dimension']=df['y_dimension'].astype('float64')
df['z_dimension']=df['z_dimension'].astype('float64')
df['depth']=df['depth'].astype('float64')
df['table']=df['table'].astype('float64')
df['price']=df['price'].astype('float64')

df.drop_duplicates()

#those two can be mean, overall size of each is less than 5% of data frame
df['price'].fillna(df['price'].mean(), inplace=True)
df['x_dimension'].fillna(df['x_dimension'].mean(), inplace=True)

#lets fill those with median
df['y_dimension'].fillna(df['y_dimension'].median(), inplace=True)
df['z_dimension'].fillna(df['z_dimension'].median(), inplace=True)
df['depth'].fillna(df['depth'].median(), inplace=True)

#and carat with most frequent
df['carat'].fillna(df['carat'].mode().iloc[0], inplace=True)
df['table'].fillna(df['table'].mode().iloc[0], inplace=True)

df['color']=df['color'].str.upper()
df['cut']=df['cut'].str.upper()
df['clarity']=df['clarity'].str.upper()

low=df['price'].quantile(0.04)
high = df['price'].quantile(0.97)
df=df[(df['price']>=low )& (df['price']<=high)]

high = df['carat'].quantile(0.99)
high = df['carat'].quantile(0.95)
df=df[(df['carat']<=high )]
st.dataframe(df.T)

plots_list=("violin","histogram","box","heatmap")
selected_plot=st.selectbox("Available Plots:",plots_list)
list_with_categories=("clarity","color","cut","carat", "x_dimension","y_dimension","z_dimension","depth", "table","price")
list_without_categories=("carat","x_dimension","y_dimension","z_dimension","depth","table","price")
list_of_categories=("clarity","color","cut","price")

if selected_plot==plots_list[0]:
    selectable_list=list_without_categories
if selected_plot==plots_list[1]:
    selectable_list=list_with_categories
if selected_plot==plots_list[2]:
    selectable_list=list_without_categories
if selected_plot==plots_list[3]:
    selectable_list=list_of_categories

selected_category=st.selectbox("Available Categories:",selectable_list)
fig, ax = plt.subplots()

df_grouped=df
for x in list_without_categories:
    if x !=selected_category:
        df_grouped[x]= (df_grouped[x]-df_grouped[x].mean())/df_grouped[x].std()

if selected_plot==plots_list[0]:
    ax.violinplot(df[selected_category])
if selected_plot==plots_list[1]:
    ax.hist(df[selected_category],bins=len(df[selected_category].unique()))
if selected_plot==plots_list[2]:
    if selected_category==list_of_categories[0]:
        df_grouped=df_grouped.drop(list_of_categories[1],axis=1)
        df_grouped=df_grouped.drop(list_of_categories[2],axis=1)
    elif selected_category==list_of_categories[1]:
        df_grouped=df_grouped.drop(list_of_categories[0],axis=1)
        df_grouped=df_grouped.drop(list_of_categories[2],axis=1)
    elif selected_category==list_of_categories[2]:
        df_grouped=df_grouped.drop(list_of_categories[0],axis=1)
        df_grouped=df_grouped.drop(list_of_categories[1],axis=1)
    else:
        df_grouped=df_grouped.drop(list_of_categories[0],axis=1)
        df_grouped=df_grouped.drop(list_of_categories[1],axis=1)
        df_grouped=df_grouped.drop(list_of_categories[2],axis=1)
    df_grouped=df_grouped.groupby([selected_category]).sum()
    df_grouped=df_grouped.div(df_grouped.sum(axis=1), axis=0)
    ax=sns.boxplot(df_grouped, showfliers = False)
if selected_plot==plots_list[3]:
    if selected_category==list_of_categories[0]:
        df_grouped=df_grouped.drop(list_of_categories[1],axis=1)
        df_grouped=df_grouped.drop(list_of_categories[2],axis=1)
    elif selected_category==list_of_categories[1]:
        df_grouped=df_grouped.drop(list_of_categories[0],axis=1)
        df_grouped=df_grouped.drop(list_of_categories[2],axis=1)
    elif selected_category==list_of_categories[2]:
        df_grouped=df_grouped.drop(list_of_categories[0],axis=1)
        df_grouped=df_grouped.drop(list_of_categories[1],axis=1)
    else:
        df_grouped=df_grouped.drop(list_of_categories[0],axis=1)
        df_grouped=df_grouped.drop(list_of_categories[1],axis=1)
        df_grouped=df_grouped.drop(list_of_categories[2],axis=1)
    if selected_category==list_of_categories[3]:
        fig, ax = plt.subplots(figsize=(5, 20))

    df_grouped=df_grouped.groupby([selected_category]).sum()
    df_grouped=df_grouped.div(df_grouped.sum(axis=1), axis=0)
    ax=sns.heatmap(df_grouped,annot=True, cmap=plt.cm.hot,vmin=-2,vmax=2, linewidths=.25)

st.pyplot(fig)