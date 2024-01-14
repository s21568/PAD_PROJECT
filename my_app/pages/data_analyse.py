import streamlit as st
import pandas as pd
import numpy as np
import st_pages as stp

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

st.header("Imported data")
df = pd.read_csv("data\messy_data.csv")
st.dataframe(df.T)

st.text("how many rows of data we have?")
st.code("len(df)")
st.text(len(df))

st.text("let`s remove all whitespace characters from columns names,\n and change x,y,z dimentions whitespaces to '_'")
st.code("""df.columns=df.columns.str.replace(' ','')
df.rename(columns={
    'xdimension':'x_dimension',
    'ydimension':'y_dimension',
    'zdimension':'z_dimension',
},inplace=True)""")
df.columns=df.columns.str.replace(' ','')
df.rename(columns={
    'xdimension':'x_dimension',
    'ydimension':'y_dimension',
    'zdimension':'z_dimension',
},inplace=True)
st.dataframe(df.T)

st.text("we can tell that clarity, color and cut will be category type, let`s check those unique values")
st.code("""df['clarity'].unique()
df['color'].unique()
df['cut'].unique()""")
st.text("clarity:")
st.table(df['clarity'].unique())
st.text("color:")
st.table(df['color'].unique())
st.text("cut:")
st.table(df['cut'].unique())

st.text(f'they seem to be not really unique, \nwe will have to fix that in next part of date preparation - {sidebar_list[2]}')

st.text("let`s check how many fields are nan")
st.code("""st.dataframe(df.isna().sum())""")
st.dataframe(df.isna().sum())
st.text("that`s strange, let`s replace empty strings with NaNs")
st.code("""df=df.replace(r'^\s*$', np.nan, regex=True)
st.dataframe(df.isna().sum())""")
df=df.replace(r'^\s*$', np.nan, regex=True)
st.dataframe(df.isna().sum())

st.text("seems more legit, lets check columns types")
st.code("""st.dataframe(df.dtypes)""")
st.dataframe(df.dtypes)

st.text("that`s not right, lets fix it")
st.code("""df['clarity']=df['clarity'].astype('category')
df['color']=df['color'].astype('category')
df['cut']=df['cut'].astype('category')
df['x_dimension']=df['x_dimension'].astype('float64')
df['y_dimension']=df['y_dimension'].astype('float64')
df['z_dimension']=df['z_dimension'].astype('float64')
df['depth']=df['depth'].astype('float64')
df['table']=df['table'].astype('float64')
df['price']=df['price'].astype('float64')""")
df['clarity']=df['clarity'].astype('category')
df['color']=df['color'].astype('category')
df['cut']=df['cut'].astype('category')
df['x_dimension']=df['x_dimension'].astype('float64')
df['y_dimension']=df['y_dimension'].astype('float64')
df['z_dimension']=df['z_dimension'].astype('float64')
df['depth']=df['depth'].astype('float64')
df['table']=df['table'].astype('float64')
df['price']=df['price'].astype('float64')
st.code("""st.dataframe(df.dtypes)""")
st.dataframe(df.dtypes)
st.header(f'next part: {sidebar_list[2]}. \nopen it by clicking {sidebar_list[2]} link')
