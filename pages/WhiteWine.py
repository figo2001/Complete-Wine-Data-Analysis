import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.stats import pearsonr
import streamlit as st

white_wine_df=pd.read_csv('winequality-white.csv',sep=';')

# For Red Wine

st.header('C) White Wine Data Exploration 🍸')

st.write(" ")

st.image("white.jpeg", width=300)
st.write(" ")

# Columns
st.subheader('White wine columns')
st.write(white_wine_df.columns)
st.write("-------")

st.subheader('Display the first five rows')
st.write(white_wine_df.head())
st.write("-------")

st.subheader('Display the last five rows')
st.write(white_wine_df.tail())
st.write("-------")

st.subheader('Shape of the data')
st.write(white_wine_df.shape)
st.write("-------")

st.subheader('Showing number of missing values')
st.write(white_wine_df.isnull().sum())
st.write("-------")

st.subheader('Showing the duplicates')
st.write(white_wine_df.duplicated().sum())
st.write("-------")

st.subheader('Showing unique columns')
st.write(white_wine_df.nunique())
st.write("-------")

st.subheader('Showing statistics')
st.write(white_wine_df.describe())
st.write("-------")

st.subheader('a) Exploration of Labeled Data') 
plt.figure(figsize=(10, 5))
sn.countplot(x='quality', data=white_wine_df, palette='viridis')
# Use Streamlit's pyplot function to display the plot
st.pyplot(plt)
st.markdown("""
- **As you can see the quality number 6 wine is available for everyone(majority).**
- **On the other hand the the quality number 9 is very less with extremely good quality and number 3 wine is the worst wine.**
""", unsafe_allow_html=True)
st.write("-------")

st.subheader('b) Correlation Matrix using Heatmap')
plt.figure(figsize=(10, 5))
sn.heatmap(white_wine_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
st.pyplot(plt)
st.markdown("""
- **Free Sulpher Dioxide and Total Sulpher Dioxide have some positive relation to Residual Sugar.On further inspection, I found that the quantity of SO2 is dependent on Sugar content.**
- **Chlorides, density and volatile acidity have weak negetive correlation with quality.**
- **alcohol has positive correlation with quality.**
""", unsafe_allow_html=True)
st.write("-------")

st.subheader("c) Visualizing the Correlations with Respect to Quality Column")
plt.figure(figsize=(20,6))
white_wine_df.corr()['quality'].plot(kind='bar')
st.pyplot(plt)
st.write("-------")

st.subheader("d) Visualizing the Distribution of Alcohol Column")
plt.figure(figsize=(10,6))
sn.histplot(white_wine_df['alcohol'], kde=True)
st.pyplot(plt)
st.markdown('**Alcohol content is positively skewed.**')
st.write("-------")

st.subheader('e) Skewness, Mean and Median')
from scipy.stats import skew
st.write('Skewness is: ',skew(white_wine_df['alcohol']))
st.write('Mean is: ',white_wine_df['alcohol'].mean())
st.write('Median is: ',white_wine_df['alcohol'].median())
st.write("-------")

st.subheader("e) Showing the relationship between Quality and Alcohol")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='alcohol', data=white_wine_df, palette='Set2')
st.pyplot(plt)
st.markdown("""
- **As quality improves, the alcohol trend is higher. So it shows a positive relation.**
- **It shows less correlation between alcohol and Quality.**
""", unsafe_allow_html=True)
correlation_coefficient, p_value = pearsonr(white_wine_df['alcohol'], white_wine_df['pH'])
st.write("Pearson correlation coefficient:", correlation_coefficient)
st.write("P-value:", p_value)
st.write("-------")

st.subheader('f) Relationship between quality and pH')
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='pH', data=white_wine_df, palette='Set3')
st.pyplot(plt)
st.write("-------")

st.subheader("f) The relationship between Alcohol and Density.")
plt.figure(figsize=(10,6))
joint_plot=sn.jointplot(x='alcohol', y='density', data=white_wine_df, kind='reg', palette='mako')
st.pyplot(plt)
st.write("**It's a negative correlation between alcohol and density**")
correlation_coefficient, p_value = pearsonr(white_wine_df['alcohol'], white_wine_df['density'])
st.write("Pearson correlation coefficient:", correlation_coefficient)
st.write("P-value:", p_value)
st.write("-------")

plt.figure(figsize=(10,6))
g=sn.FacetGrid(white_wine_df, col='quality')
g=g.map(sn.regplot, 'pH','alcohol')
st.pyplot(plt)
st.write("-------")

st.subheader("g) Let's analyze sulphates and quality")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='sulphates', data=white_wine_df, palette='mako')
st.pyplot(plt)
st.write("-------")

st.subheader(" h) The relation bewtween quality and total sulfur dioxide")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='total sulfur dioxide', data=white_wine_df, palette='magma')
st.pyplot(plt)
correlation_coefficient, p_value = pearsonr(white_wine_df['quality'], white_wine_df['total sulfur dioxide'])
st.write("Pearson correlation coefficient:", correlation_coefficient)
st.write("P-value:", p_value)
st.markdown("This tends to be a weak correlation.")
st.write("-------")

st.subheader(" i) The relation bewtween quality and free sulfur dioxide")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='free sulfur dioxide', data=white_wine_df, palette='husl')
st.pyplot(plt)
correlation_coefficient, p_value = pearsonr(white_wine_df['quality'], white_wine_df['free sulfur dioxide'])
st.write("Pearson correlation coefficient:", correlation_coefficient)
st.write("P-value:", p_value)
st.markdown("**There's not much correlation present in here.**")
st.write("-------")

st.subheader(" j) The relation bewtween quality and Volatile Acidity")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='volatile acidity', data=white_wine_df, palette='colorblind')
st.pyplot(plt)
correlation_coefficient, p_value = pearsonr(white_wine_df['quality'], white_wine_df['volatile acidity'])
st.write("Pearson correlation coefficient:", correlation_coefficient)
st.write("P-value:", p_value)
st.markdown("**There's weak negetive correlation present in here.**")
st.write("-------")

st.subheader("k) Relation between Residual Sugar and Density")
plt.figure(figsize=(10,6))
joint_plot=sn.jointplot(x='residual sugar', y='density', data=white_wine_df, kind='reg')
st.pyplot(plt)
st.markdown("**There's high correlation present in here.**")
st.write("-------")

st.subheader(" l) Create a new column total acidity")
white_wine_df['total acidity']=(white_wine_df['fixed acidity']+white_wine_df['citric acid']+ white_wine_df['volatile acidity'])
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='total acidity', data=white_wine_df, palette='Set1')
st.pyplot(plt)
st.markdown("**There's no such correlation present in here.**")
correlation_coefficient, p_value = pearsonr(white_wine_df['quality'], white_wine_df['total acidity'])
st.write("Pearson correlation coefficient:", correlation_coefficient)
st.write("P-value:", p_value)
st.write("-------")

st.subheader("m) Let's move on to citric acid")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='citric acid', data=white_wine_df, palette='Set3')
st.pyplot(plt)
st.write("-------")

st.subheader("n) Creating a joint Plot of pH and citric acid.")
plt.figure(figsize=(10,6))
joint_plot=sn.jointplot(x='pH', y='citric acid', data=white_wine_df, kind='reg')
st.pyplot(plt)
correlation_coefficient, p_value = pearsonr(white_wine_df['pH'], white_wine_df['citric acid'])
st.write("Pearson correlation coefficient:", correlation_coefficient)
st.write("P-value:", p_value)
st.markdown("**It's's showing a very low correlation in here.**")
st.write("-------")

st.subheader("o) Finally let's check Relation between Residual Sugar and Quality")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='residual sugar', data=white_wine_df, palette='Set2')
st.pyplot(plt)
st.write("-------")

st.subheader("p) Creating a new column Crisp Ratio")
white_wine_df['Crisp Ratio']=white_wine_df['total acidity'] / white_wine_df['residual sugar']
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='Crisp Ratio', data=white_wine_df, showfliers=False, palette='dark')
st.pyplot(plt)
st.write("-------")

st.subheader("After analyszing the White wine, we can say that:")
st.markdown("""
- **This is not a sweet wine.**
- **There is good amount of acidity present.**
- **Total acidity is actually overpowering the residual sugar.**
""", unsafe_allow_html=True)

st.write("-------")
