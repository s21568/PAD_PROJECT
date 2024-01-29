import streamlit as st
import pandas as pd
import numpy as np
import st_pages as stp

sidebar_list=("Dashboard",
              "Data analysys",
              "Data cleanup",
              "Data visualisation",
              "Regresion model preparation")
stp.add_page_title()
stp.show_pages([
    stp.Page("my_app\dashboard.py", sidebar_list[0],icon= ":notebook:"),
    stp.Section(name="Data preparation", icon=":books:"),
    stp.Page("my_app\pages\data_analyse.py", sidebar_list[1], icon=":books:",in_section=True),
    stp.Page("my_app\pages\data_cleanup.py", sidebar_list[2],icon= ":books:",in_section=True),
    stp.Page("my_app\pages\data_visualization.py", sidebar_list[3], icon=":bar_chart:",in_section=True),
    stp.Section(name="Regression model", icon=":bar_chart:"),
    stp.Page("my_app\pages\\regresion_model_preparation.py", sidebar_list[4],icon= ":books:",in_section=True)
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
st.dataframe(df.T)
st.divider()
st.text("lets remove all duplicates")
st.code("""pre_duplicate_removal_size =len(df)
df.drop_duplicates()
print(f'removed {len(df)-pre_duplicate_removal_size} duplicated rows')""")
pre_duplicate_removal_size =len(df)
df.drop_duplicates()
st.text(f'removed {len(df)-pre_duplicate_removal_size} duplicated rows')

st.text("no duplicates, that good I suppose. now lets fill those NaNs")
st.code("""df.isna().sum()""")
st.text(df.isna().sum())
st.code("""#those two can be mean, overall size of each is less than 5% of data frame
df['price'].fillna(df['price'].mean(), inplace=True)
df['x_dimension'].fillna(df['x_dimension'].mean(), inplace=True)

#lets fil those with median
df['y_dimension'].fillna(df['y_dimension'].median(), inplace=True)
df['z_dimension'].fillna(df['z_dimension'].median(), inplace=True)
df['depth'].fillna(df['depth'].median(), inplace=True)

#and carat with most frequent
df['carat'].fillna(df['carat'].mode().iloc[0], inplace=True)
df['table'].fillna(df['table'].mode().iloc[0], inplace=True)
df.isna().sum()""")
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
st.text(df.isna().sum())
st.divider()

st.text("move all category names to upper, thus removing human typing errors")
st.code("""df['color']=df['color'].str.upper()
df['cut']=df['cut'].str.upper()
df['clarity']=df['clarity'].str.upper()
df""")
df['color']=df['color'].str.upper()
df['cut']=df['cut'].str.upper()
df['clarity']=df['clarity'].str.upper()
st.text("clarity:")
st.table(df['clarity'].unique())
st.text("color:")
st.table(df['color'].unique())
st.text("cut:")
st.table(df['cut'].unique())
st.divider()
st.text("now let`s translate color colorless to mean of color values(acoring to standarization)")
st.code("df['color'].replace('COLORLESS','E',inplace=True,regex=True)")
df['color'].replace('COLORLESS','E',inplace=True,regex=True)
st.text("now let`s translate categories to numeric - categories scale from internet forums")
df['clarity'].replace('FL',11,inplace=True,regex=True)
df['clarity'].replace('IF',10,inplace=True,regex=True)
df['clarity'].replace('VVS2',9,inplace=True,regex=True)
df['clarity'].replace('VVS1',8,inplace=True,regex=True)
df['clarity'].replace('VS2',7,inplace=True,regex=True)
df['clarity'].replace('VS1',6,inplace=True,regex=True)
df['clarity'].replace('SI2',5,inplace=True,regex=True)
df['clarity'].replace('SI1',4,inplace=True,regex=True)
df['clarity'].replace('I1',3,inplace=True,regex=True)
df['clarity'].replace('I2',2,inplace=True,regex=True)
df['clarity'].replace('I3',1,inplace=True,regex=True)

df['color'].replace('D',8,inplace=True,regex=True)
df['color'].replace('E',7,inplace=True,regex=True)
df['color'].replace('F',6,inplace=True,regex=True)
df['color'].replace('G',5,inplace=True,regex=True)
df['color'].replace('H',4,inplace=True,regex=True)
df['color'].replace('I',3,inplace=True,regex=True)
df['color'].replace('J',2,inplace=True,regex=True)
df['color'].replace('K',1,inplace=True,regex=True)

df['cut'].replace('IDEAL',5,inplace=True,regex=True)
df['cut'].replace('PREMIUM',4,inplace=True,regex=True)
df['cut'].replace('GOOD',3,inplace=True,regex=True)
df['cut'].replace('FAIR',2,inplace=True,regex=True)
df['cut'].replace('	VERY GOOD',1,inplace=True,regex=True)

st.code("""df['clarity'].replace('FL',11,inplace=True,regex=True)
df['clarity'].replace('IF',10,inplace=True,regex=True)
df['clarity'].replace('VVS2',9,inplace=True,regex=True)
df['clarity'].replace('VVS1',8,inplace=True,regex=True)
df['clarity'].replace('VS2',7,inplace=True,regex=True)
df['clarity'].replace('VS1',6,inplace=True,regex=True)
df['clarity'].replace('SI2',5,inplace=True,regex=True)
df['clarity'].replace('SI1',4,inplace=True,regex=True)
df['clarity'].replace('I1',3,inplace=True,regex=True)
df['clarity'].replace('I2',2,inplace=True,regex=True)
df['clarity'].replace('I3',1,inplace=True,regex=True)

df['color'].replace('D',8,inplace=True,regex=True)
df['color'].replace('E',7,inplace=True,regex=True)
df['color'].replace('F',6,inplace=True,regex=True)
df['color'].replace('G',5,inplace=True,regex=True)
df['color'].replace('H',4,inplace=True,regex=True)
df['color'].replace('I',3,inplace=True,regex=True)
df['color'].replace('J',2,inplace=True,regex=True)
df['color'].replace('K',1,inplace=True,regex=True)
        
df['cut'].replace('IDEAL',5,inplace=True,regex=True)
df['cut'].replace('PREMIUM',4,inplace=True,regex=True)
df['cut'].replace('GOOD',3,inplace=True,regex=True)
df['cut'].replace('FAIR',2,inplace=True,regex=True)
df['cut'].replace('VERY GOOD',1,inplace=True,regex=True)""")
st.divider()
st.text("and lets change data type of those columns to numeric")
st.code("""df['clarity']=df['clarity'].astype('float64')
df['color']=df['color'].astype('float64')
df['cut']=df['cut'].astype('float64')""")
df['clarity']=df['clarity'].astype('float64')
df['color']=df['color'].astype('float64')
df['cut']=df['cut'].astype('float64')
st.dataframe(df.T)

st.divider()
st.text("now lets check for outliers")
st.code("""df['carat'].describe()""")
st.dataframe(df['carat'].describe())
st.code("""df['x_dimension'].describe()""")
st.dataframe(df['x_dimension'].describe())
st.code("""df['y_dimension'].describe()""")
st.dataframe(df['y_dimension'].describe())
st.code("""df['z_dimension'].describe()""")
st.dataframe(df['z_dimension'].describe())
st.code("""df['price'].describe()""")
st.dataframe(df['price'].describe())

st.text("price and carat should be cleared of outliers, especially price. \nlets check if price quantile range of 1 to 99 will do the job ")
st.code("""low=df['price'].quantile(0.01)
high = df['price'].quantile(0.99)""")
low=df['price'].quantile(0.01)
high = df['price'].quantile(0.99)
st.text(f'quantiles of 1 = {low} and 99 = {high}')
st.text('nope, lets go for 5 and 97')
st.code("""low=df['price'].quantile(0.04)
high = df['price'].quantile(0.97)""")
low=df['price'].quantile(0.04)
high = df['price'].quantile(0.97)
st.text(f'quantiles of 4 = {low} and 97 = {high}')
st.text('seems better, lets filter out data')
st.code("""df=df[(df['price']>=low )& (df['price']<=high)]
df['price'].describe()""")
df=df[(df['price']>=low )& (df['price']<=high)]
st.dataframe(df['price'].describe())

st.text("now carats")
st.code("""high = df['carat'].quantile(0.99)
print(f'quantiles 99 = {high}')
print('nope lets go for 95')
high = df['carat'].quantile(0.95)
print(f'quantiles 95 = {high}')
df=df[(df['carat']<=high )]
df['carat'].describe()""")
high = df['carat'].quantile(0.99)
st.text(f'quantiles 99 = {high}')
st.text('nope lets go for 95')
high = df['carat'].quantile(0.95)
st.text(f'quantiles 95 = {high}')
df=df[(df['carat']<=high )]
st.dataframe(df['carat'].describe())

st.divider()
st.text("how many records do we have now(199 at the begining)?")
st.code("""len(df)""")
st.text(len(df))
st.text("lets stay there, difference is not that big yet")
st.dataframe(df.T)