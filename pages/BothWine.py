import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import streamlit as st

st.header("A) Comparative Analysis of White Wine and Red Wine Data 🍸🍷")
st.write(" ")

red_wine_df = pd.read_csv('winequality-red.csv', sep=';')
white_wine_df=pd.read_csv('winequality-white.csv',sep=';')

st.subheader("Show statisical measures of red wine")
st.write(red_wine_df.describe())

st.subheader("Show statisical measures of white wine")
st.write(white_wine_df.describe())

st.markdown("""
- **Residual sugars are comparatively higher in White Wines compare to Red Wines.**
- **Sulpher Di-oxide are comparatively higher in White Wines compare to Red Wines.**
- **Densiy is more or less same in white wine and red wine.**
- **pH,sulphates,alcohol quantity is kind of same of both wines.**
- **Same goes for Quality.**
""", unsafe_allow_html=True)
st.write("-------")

st.header(" Now, let's talking about Correlations")
st.subheader(" A) For Red wine: ")
plt.figure(figsize=(10,6))
sn.heatmap(red_wine_df.corr(), annot=True, cmap='viridis',fmt='.2f')
st.pyplot(plt)
st.markdown("""
### i) For Red wine, with respect to Quality column:

#### a) For Positive Correlation:
    1. Alcohol
    2. Fixed Acidity
    3. Sulphates
    4. Citric Acid
    
#### b) For Negative Correlation:
    1. Volatile Acidity
    2. Total Sulfur dioxide
    3. Density
    4. Chlorides
""")
st.subheader(" B) For White wine: ")
plt.figure(figsize=(10,6))
sn.heatmap(white_wine_df.corr(), annot=True, cmap='viridis',fmt='.2f')
st.pyplot(plt)
st.markdown("""
### ii) For White wine, with respect to Quality column:

#### a) For Positive Correlation:
    1. Alcohol
    2. pH(weak)
   
#### b) For Negative Correlation:
    1. Volatile Acidity
    2. Total Sulfur dioxide
    3. Density
    4. Chlorides
    5. residual sugar(weak)
""")
st.write("-------")

st.subheader("Combine the datasets")
red_wine_df['type']='Red'
white_wine_df['type']='White'
wines_df=pd.concat([red_wine_df, white_wine_df])

st.markdown("**After Merged, the first five rows of the Dataframe named wines_df**")
st.write(wines_df.head())
st.markdown("**The last five rows.**")
st.write(wines_df.tail())
st.write("-------")

st.markdown("The shape of the dataset")
st.write(wines_df.shape)
st.write("-------")

st.header("Comparative Analysis 📈📊")

st.write(" ")

st.subheader("C) For the Quality Column")
plt.figure(figsize=(10,6))
sn.countplot(x='quality', hue='type', data=wines_df)
st.pyplot(plt)
st.markdown("""
- **If you observe the dataset, you can see that the number of samples are more in White wine with respect to Red wine.**
- **So the Plot is looks like this.**
- **And WIne Quality 9 is not present in Red wine. It only present in White Wines.**
""", unsafe_allow_html=True)
st.write("-------")

st.subheader("D) PLotting a Density Plot for more clarity")
plt.figure(figsize=(10,6))
p1=sn.kdeplot(red_wine_df['quality'], shade=True, color='r', label='red whine')
p2=sn.kdeplot(white_wine_df['quality'], shade=True, color='b', label='white whine')
plt.legend()
st.pyplot(plt)
st.markdown("**For the size of the white wine it's overlapping the graph for red wine.**")
st.write("-------")

st.subheader("E) Relation with quality and alcohol Column")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='alcohol', hue='type', data=wines_df, palette=['r','w'])
st.pyplot(plt)
st.markdown("""
- **Overall alcohol content seems to little higher for white wine comparing with red wine.**
- **Except for Quality 8, where Red wine is slightly higher.**
""", unsafe_allow_html=True)
st.write("-------")

st.subheader("Now for the Density Column")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='density', hue='type', data=wines_df, palette=['r','w'], showfliers=False)
st.pyplot(plt)
st.markdown("""
- **It actually shows us a negetive correlation means, if quality improves then density will be lower.**
- **Also Red wine has more density than white wines.**
""", unsafe_allow_html=True)
st.write("-------")

st.subheader("F) PLotting a Joint plot with alcohol and Residual sugar")
plt.figure(figsize=(10,6))
sn.jointplot(x='alcohol', y='residual sugar', data=wines_df, hue='type')
st.pyplot(plt)
st.write("-------")

st.subheader("Now Visualizing using Boxplot to find any relation")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='residual sugar', hue='type', data=wines_df, palette=['r','w'], showfliers=False)
st.pyplot(plt)
st.markdown("**So as you can see red wine has very less sugar as compared to White wines.**")
st.write("-------")

st.subheader("G) The Distributions of Residual Sugar for Both Wines")
plt.figure(figsize=(10,6))
p1=sn.kdeplot(red_wine_df['residual sugar'], shade=True, color='r', label='red whine')
p2=sn.kdeplot(white_wine_df['residual sugar'], shade=True, color='b', label='white whine')
plt.legend()
st.pyplot(plt)
st.write("-------")

st.subheader("Visualizing the relation between alcohol and residual sugar")
plt.figure(figsize=(10,6))
sn.regplot(x='alcohol', y='residual sugar', data=wines_df)
st.pyplot(plt)
st.markdown("**So as you can see that it showing us a negetive relation between The Alcohol and Residual Sugar.**")
st.write("-------")

st.subheader("H) Next we will analyzing the Sulphates Columns")

st.subheader("i) For Total Sulfur Dioxide")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='total sulfur dioxide', hue='type', data=wines_df, palette=['r','w'])
st.pyplot(plt)
st.write("-------")

st.subheader("ii) For Free Sulfur Dioxide")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='free sulfur dioxide', hue='type', data=wines_df, palette=['r','w'])
st.pyplot(plt)
st.markdown("**So as you can see that there is so much difference betweem Red wine and White wine.**")
st.write("-------")

st.subheader("iii) Now with Sulphates")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='sulphates', hue='type', data=wines_df, palette=['r','w'],showfliers=False)
st.pyplot(plt)
st.markdown("""
- **As you can see the quantity of sulphates is greater in Red wine.**
- **AS it compare to white wine as you can see there is a lesser amount of sulpher di-oxide. That's why it a lesser than Red Wine.**
""", unsafe_allow_html=True)
st.write("-------")

st.subheader("I) Now Comparing with Citric Acid")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='citric acid', hue='type', data=wines_df, palette=['r','w'],showfliers=False)
st.pyplot(plt)
st.markdown("**It is a positive relationship, that we are seeing in here.**")
st.write("-------")

st.subheader(" Now for the Clorides")
plt.figure(figsize=(10,6))
sn.boxplot(x='quality', y='chlorides', hue='type', data=wines_df, palette=['r','w'],showfliers=False)
st.pyplot(plt)
st.markdown("**As you can see it describes here as a negetive relationship.**")
st.write("-------")

st.subheader("J) Now we are Combining those three acidities and make a new column called Total acidity")
plt.figure(figsize=(10,6))
wines_df['total acidity']=wines_df['fixed acidity'] + wines_df['volatile acidity'] + wines_df['citric acid']
sn.boxplot(x='quality', y='total acidity', hue='type', data=wines_df,
          palette=['r','w'], showfliers=False)
st.pyplot(plt)
st.write("-------")

st.subheader("Conclusion")
st.markdown("""
- **Red Wine seem to be overall more acidic as compare to White Wines.**
- **his might be cause that Red wines have a presence of a distinctive component in red wines known as tannins, , that basically         combines acidity and it brings unique test to Red Wine. And that is not present in White Wine.**
""", unsafe_allow_html=True)
st.write("-------")
