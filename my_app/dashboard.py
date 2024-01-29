import numpy as np
import pandas as pd
import streamlit as st
import st_pages as stp
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

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

df['clarity'].replace('FL',1,inplace=True,regex=True)
df['clarity'].replace('IF',2,inplace=True,regex=True)
df['clarity'].replace('VVS2',3,inplace=True,regex=True)
df['clarity'].replace('VVS1',4,inplace=True,regex=True)
df['clarity'].replace('VS2',5,inplace=True,regex=True)
df['clarity'].replace('VS1',6,inplace=True,regex=True)
df['clarity'].replace('SI2',7,inplace=True,regex=True)
df['clarity'].replace('SI1',8,inplace=True,regex=True)
df['clarity'].replace('I1',9,inplace=True,regex=True)
df['clarity'].replace('I2',10,inplace=True,regex=True)
df['clarity'].replace('I3',11,inplace=True,regex=True)

df['color'].replace('COLORLESS','E',inplace=True,regex=True)
df['color'].replace('D',1,inplace=True,regex=True)
df['color'].replace('E',2,inplace=True,regex=True)
df['color'].replace('F',3,inplace=True,regex=True)
df['color'].replace('G',4,inplace=True,regex=True)
df['color'].replace('H',5,inplace=True,regex=True)
df['color'].replace('I',6,inplace=True,regex=True)
df['color'].replace('J',7,inplace=True,regex=True)
df['color'].replace('K',8,inplace=True,regex=True)

df['cut'].replace('IDEAL',1,inplace=True,regex=True)
df['cut'].replace('PREMIUM',2,inplace=True,regex=True)
df['cut'].replace('GOOD',3,inplace=True,regex=True)
df['cut'].replace('FAIR',4,inplace=True,regex=True)
df['cut'].replace('	VERY GOOD',5,inplace=True,regex=True)
df['clarity']=df['clarity'].astype('float64')
df['color']=df['color'].astype('float64')
df['cut']=df['cut'].astype('float64')

low=df['price'].quantile(0.04)
high = df['price'].quantile(0.97)
df=df[(df['price']>=low )& (df['price']<=high)]

high = df['carat'].quantile(0.99)
high = df['carat'].quantile(0.95)
df=df[(df['carat']<=high )]
st.dataframe(df.T)
st.divider()
plots_checkbox=st.checkbox("Show plots",key="plots_checkbox")
plots_list=("violin","histogram","box","heatmap")
list_with_categories=("clarity","color","cut","carat", "x_dimension","y_dimension","z_dimension","depth", "table","price")
list_of_categories=("clarity","color","cut","price")
fig, ax = plt.subplots()
if plots_checkbox:
    selected_plot=st.selectbox("Available Plots:",plots_list)
    selected_category=st.selectbox("Available Categories:",list_with_categories)
    df_grouped=df
    for x in list_with_categories:
        if x !=selected_category:
            df_grouped[x]= (df_grouped[x]-df_grouped[x].mean())/df_grouped[x].std()

    if selected_plot==plots_list[0]:
        ax.violinplot(df[selected_category])
    if selected_plot==plots_list[1]:
        ax.hist(df[selected_category],bins=len(df[selected_category].unique()))
    if selected_plot==plots_list[2]:
        df_grouped=df_grouped.groupby([selected_category]).sum()
        df_grouped=df_grouped.div(df_grouped.sum(axis=1), axis=0)
        ax=sns.boxplot(df_grouped, showfliers = False)
    if selected_plot==plots_list[3]:
        if selected_category==list_of_categories[3]:
            fig, ax = plt.subplots(figsize=(5, 20))

        df_grouped=df_grouped.groupby([selected_category]).sum()
        df_grouped=df_grouped.div(df_grouped.sum(axis=1), axis=0)
        ax=sns.heatmap(df_grouped,annot=True, cmap=plt.cm.hot,vmin=-2,vmax=2, linewidths=.25)
    st.pyplot(fig)
st.divider()
single_category_checkbox=st.checkbox("Show single category impact",key="single_category_checkbox")
if single_category_checkbox:
    selected_category=st.selectbox("Available Categories:",list_with_categories)
    stats_checkbox=st.checkbox('Show stats')
    diagram_checkbox=st.checkbox('Show diagram')

    df_fitted=df
    for x in list_with_categories:
        df_fitted[x]= (df_fitted[x]-df_fitted[x].mean())/df_fitted[x].std()
    model = smf.ols(formula="price ~ "+selected_category,data= df).fit()
    df_fitted['fitted']=model.fittedvalues
    if stats_checkbox:
        st.text(model.summary())
    if diagram_checkbox:
        fig, ax = plt.subplots()
        ax.scatter( x=df_fitted[selected_category], y=df_fitted['price'])
        ax.scatter( x=df_fitted[selected_category], y=df_fitted['fitted'])
    st.pyplot(fig)
st.divider()
multiple_categories_checkbox=st.checkbox("Show  multiple categories impact",key="multiple_categories_checkbox")
if multiple_categories_checkbox:
    selected_categories=st.multiselect("Available Categories:",list_with_categories)
    stats_checkbox_categories=st.checkbox('Show stats',key="second")
    formula_string="price ~ "
    if selected_categories:
        for x in selected_categories:
            if x.index == 0:
                formula_string=formula_string+" "+x
            else:
                formula_string=formula_string+ " + "+x
        model = smf.ols(formula_string,data= df).fit()
        df_fitted['fitted']=model.fittedvalues
        if stats_checkbox_categories:
            st.text(model.summary())
        corelation_checkbox_categories_2=st.checkbox('Show categories correlation',key="corr_categories")
        if corelation_checkbox_categories_2:
            corelation_heatmap_checkbox_categories=st.checkbox('Show categories correlation heatmap',key="corr_heatmap_2")
            df_corr=df[df.columns.intersection(list_with_categories).intersection(selected_categories)]
            st.dataframe(df_corr.corr())
            st.text('ssr: ')
            st.text(model.ssr)
            if corelation_heatmap_checkbox_categories:
                ax.imshow(df_corr.corr(),cmap=plt.cm.hot)
                st.pyplot(fig)
st.divider()
author_categories_checkbox=st.checkbox("Show categories selected by author impact",key="author_categories_checkbox")
if author_categories_checkbox:
    selected_list=("clarity", "x_dimension","y_dimension","carat","table")
    df_selected=df[df.columns.intersection(selected_list)]
    for x in selected_list:
        df_selected[x]= (df_selected[x]-df_selected[x].mean())/df_selected[x].std()
    st.dataframe(df_selected.corr())
    formula_string="price ~ "
    for x in selected_list:
        if x.index == 0:
            formula_string=formula_string+" "+x
        else:
            formula_string=formula_string+ " + "+x
    model = smf.ols(formula_string,data= df).fit()
    st.text(model.summary())
    st.text('ssr: ')
    st.text(model.ssr)
st.divider()