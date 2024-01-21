import numpy as np
import pandas as pd
import streamlit as st
import st_pages as stp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statsmodels.formula.api as smf

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
list_of_normalized_categories=("carat", "x_dimension","y_dimension","z_dimension","depth", "table","price")
list_with_categories=("clarity","color","cut","carat", "x_dimension","y_dimension","z_dimension","depth", "table")

st.text("Price vs selected category diagram and stats")
selected_category=st.selectbox("Available Categories:",list_with_categories)
stats_checkbox=st.checkbox('Show stats')
diagram_checkbox=st.checkbox('Show diagram')

df_fitted=df
for x in list_of_normalized_categories:
    if x !=selected_category:
        df_fitted[x]= (df_fitted[x]-df_fitted[x].mean())/df_fitted[x].std()
model = smf.ols(formula="price ~ "+selected_category,data= df).fit()
df_fitted['fitted']=model.fittedvalues
df_fitted=df_fitted.sort_values('fitted')
if stats_checkbox:
    st.text(model.summary())
if diagram_checkbox:
    fig, ax = plt.subplots()
    ax.scatter( x=df_fitted[selected_category], y=df_fitted['price'])
    ax.scatter( x=df_fitted[selected_category], y=df_fitted['fitted'])
    st.pyplot(fig)

st.text("Price vs muliselect categories diagram and stats")
selected_categories=st.multiselect("Available Categories:",list_with_categories)
stats_checkbox_categories=st.checkbox('Show stats',key="second")
r2_checkbox_categories=st.checkbox('Show R^2')
formula_string="price ~ "
if selected_categories:
    for x in selected_categories:
        if x.index == 0:
            formula_string=formula_string+" "+x
        else:
            formula_string=formula_string+ " + "+x
    model = smf.ols(formula=formula_string,data= df).fit()
    df_fitted['fitted']=model.fittedvalues
    df_fitted=df_fitted.sort_values('fitted')
    if stats_checkbox_categories:
        st.text(model.summary())
    if r2_checkbox_categories:
        model_mean=smf.ols("price ~ 1",data = df_fitted).fit()
        resid_mean,resid_model=np.mean(model_mean.resid**2),np.mean(model.resid**2)
        st.text((resid_mean - resid_model)/resid_mean)