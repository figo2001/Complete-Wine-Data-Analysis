import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import streamlit as st


# Load the data
red_wine = pd.read_csv('winequality-red.csv', sep=';')


# For Red Wine

st.header('B) Red Wine Data Exploration 🍷')
st.write(" ")

st.image("red.jpg",width=300)
st.write(" ")

# Columns
st.subheader('Red wine columns')
st.write(red_wine.columns)
st.write("-------")

st.subheader('Display the first five rows')
st.write(red_wine.head())
st.write("-------")

st.subheader('Display the last five rows')
st.write(red_wine.tail())
st.write("-------")

st.subheader('Shape of the data')
st.write(red_wine.shape)
st.write("-------")

st.subheader('Showing number of missing values')
st.write(red_wine.isnull().sum())
st.write("-------")

st.subheader('Showing the duplicates')
st.write(red_wine.duplicated().sum())
st.write("-------")

st.subheader('Showing statistics')
st.write(red_wine.describe())
st.write("-------")

st.subheader('a) Exploration of Labeled Data') 
plt.figure(figsize=(10, 5))
sn.countplot(x='quality', data=red_wine, palette='viridis')
# Use Streamlit's pyplot function to display the plot
st.pyplot(plt)
st.write("-------")

st.subheader('b) Correlation Matrix using Heatmap')
plt.figure(figsize=(12, 8))
sn.heatmap(red_wine.corr(), annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot(plt)
st.write("-------")

st.write('Observations')
st.markdown("""
- **Free Sulphur Dioxide and Total Sulphur Dioxide have some positive relation to Residual Sugar.**
- **Density has a positive correlation with fixed acidity and residual sugar.**
- **Density has a negative correlation with alcohol and pH.**
- **Quality has a positive correlation with alcohol, citric acid, and sulphates and a negative correlation with citric acid. We need to explore this further.**
- **Fixed acidity has a high positive correlation with citric acid and density and a negative correlation with pH.**
- **Residual sugar has a positive correlation with citric acid.**
- **pH has a negative correlation with fixed acidity and citric acid, but a positive correlation with volatile acid.**
""", unsafe_allow_html=True)
st.write("-------")

st.subheader('c) Visualizing the co-relations with the respect to labeled column')
plt.figure(figsize=(15, 8))
red_wine.corr()['quality'].plot(kind='bar')
st.pyplot(plt)
st.write("-------")

st.subheader('d) Showing the Distribution of Alcohol Column')
plt.figure(figsize=(10,6))
sn.histplot(red_wine['alcohol'], kde=True, palette='mako')
st.pyplot(plt)
st.write('**As you can see here that, Alcohol content is positively Skewed**')
st.subheader('e) Skewness, Mean and Median')
from scipy.stats import skew
st.write('Skewness is: ',skew(red_wine['alcohol']))
st.write('Mean is: ',red_wine['alcohol'].mean())
st.write('Median is: ',red_wine['alcohol'].median())
st.write("-------")

st.subheader("f) Let's see how alcohol varies w.r.t quality")
st.markdown('**To not showing the outliers we use here showfliers=False**')
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='alcohol', data=red_wine, showfliers=False, palette='dark')
st.pyplot(plt)
st.write("-------")

st.subheader('g) Correlation with Alcohol and pH')
plt.figure(figsize=(10,6))
sn.jointplot(x='alcohol', y='pH', data=red_wine, kind='reg')
st.pyplot(plt)
st.write("**It's a postive correlation**")
from scipy.stats import pearsonr
correlation_coefficient, p_value = pearsonr(red_wine['alcohol'], red_wine['pH'])
st.write("Pearson correlation coefficient:", correlation_coefficient)
st.write("P-value:", p_value)
st.write("-------")

st.subheader('h) Co-relation with alcohol and density')
plt.figure(figsize=(10,6))
sn.jointplot(x='alcohol', y='density', data=red_wine, kind='reg')
st.pyplot(plt)
st.write("**As you can see, It's a negetive co-relation.**")
correlation_coefficient, p_value = pearsonr(red_wine['alcohol'], red_wine['density'])
st.write("Pearson correlation coefficient:", correlation_coefficient)
st.write("P-value:", p_value)
plt.figure(figsize=(10,6))
g=sn.FacetGrid(red_wine, col='quality')
g=g.map(sn.regplot, 'density','alcohol')
st.pyplot(plt)
st.markdown('**When we going to increase the quality of the wine, you can see that the correlation between the alcohol and the density is tend to negetive.**')
st.write("-------")

st.subheader("i) Let's Analyze sulphates and Quality")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='sulphates', data=red_wine, showfliers=False, palette='magma')
st.pyplot(plt)
st.markdown("**As you can see that, as the Quality improves the sulphates is going higher -- so it's indicates a positive Correlation.**")
st.write("-------")

st.subheader("i1) Let's Analyze Total Sulfur Dioxide and Quality")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='total sulfur dioxide', data=red_wine, showfliers=False, palette='colorblind')
st.pyplot(plt)
st.write("-------")

st.subheader("i2) Let's Analyze Free Sulfur Dioxide and Quality")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='free sulfur dioxide', data=red_wine, showfliers=False, palette='colorblind')
st.pyplot(plt)
st.write("-------")

st.subheader("j) Let's move on to fixed acidity, volatile acidity and critic acid")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='fixed acidity', data=red_wine, palette='Set2')
st.pyplot(plt)
st.write("-------")

plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='citric acid', data=red_wine, palette='husl')
st.pyplot(plt)
st.markdown("**It denotes the positive relationship with Quality. The more citric acid, the wine will be taste better.**")
st.write("-------")

plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='volatile acidity', data=red_wine, palette='rainbow')
st.pyplot(plt)
st.markdown('**It denoets the negetive relation. The higher the volatile acidity, the wine will be taste worst**')
st.write("-------")

st.header("Trends between other columns")

st.subheader("k) Visualizing the Correlation of ph and Volatile Acidity")
correlation_coefficient, p_value = pearsonr(red_wine['pH'], red_wine['volatile acidity'])
st.write("Pearson correlation coefficient:", correlation_coefficient)
st.write("P-value:", p_value)
st.markdown("**As you can see it showing the weaker correlation with pH. Volatile acidity is actually Acidic acid, it is weak.**")
st.write("-------")

st.subheader("l) Create a new column Total Acidity")
red_wine['total acidity']=(red_wine['fixed acidity']+ red_wine['citric acid']+ red_wine['volatile acidity'])
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='total acidity', data=red_wine, palette='mako')
st.pyplot(plt)
st.markdown('**This is actually not find any trend or relation here.**')
st.write("-------")

st.subheader('**m) The relation between pH and Total Acidity.**')
plt.figure(figsize=(10,6))
sn.regplot(x='pH', y='total acidity', data=red_wine)
st.pyplot(plt)
st.markdown('**It shows a negetive correlation. It means that if you have more acid then the ph will be lower.**')
st.write("-------")

g=sn.FacetGrid(red_wine, col='quality')
g=g.map(sn.regplot, 'total acidity','pH')
st.pyplot(plt)
st.markdown("""
- **For the Higher Quality Wines the negetive correlation is much stronger than the lower qualities wine.**
- **It also makes sense that in the lower quality the samples will be lesser.**
""", unsafe_allow_html=True)
correlation_coefficient, p_value = pearsonr(red_wine['pH'], red_wine['total acidity'])
st.write("Pearson correlation coefficient:", correlation_coefficient)
st.write("P-value:", p_value)
st.write("-------")



