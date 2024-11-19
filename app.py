
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px


st.title("World Happiness Dashboard")
st.write("Explore insights from the World Happiness Report dataset with interactive charts.")

# Load the data
df = pd.read_csv('/Users/mona/Desktop/Northeastern/1st Semester/6600/Project/Project-02/2019.csv')

# Sidebar filters
st.sidebar.header("Filters")

score_min, score_max = st.sidebar.slider(
    "Select Score Range", 
    float(df["Score"].min()), 
    float(df["Score"].max()), 
    (6.0, 8.0))

# Filter the data based on score (score of happiness)
filtered_df = df[(df["Score"] >= score_min) & (df["Score"] <= score_max)]

# Dropdown for country selection
selected_country = st.sidebar.selectbox("Select a Country", df["Country or region"].unique())

# Checkbox for showing top countries by GDP per capita
show_top_gdp = st.sidebar.checkbox("Show Top 10 Countries by GDP per capita")

# Display data for selected country
st.write(f"### Data for {selected_country}")
country_data = df[df["Country or region"] == selected_country]
st.write(country_data)



# Checkbox Score of Top 10 Countries
if st.checkbox("Show Top 10 Countries by Score"):
    filtered_df_01 = df.nlargest(10, "Score")
else:
    filtered_df_01 = df

# 1. # Bar Chart: Score by Country 

st.write('### Matplotlib Bar Chart: Score by Country')
fig, ax = plt.subplots(figsize=(12, 15))  
sns.barplot(y="Country or region", x="Score", data=country_data, ax=ax)
#sns.barplot(y="Country or region", x="Score", data=df, ax=ax)

ax.set_title("Score by Country", fontsize=16)
ax.set_xlabel("Score", fontsize=14)
ax.set_ylabel("Country", fontsize=14)

#set fonts
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)   

st.pyplot(fig)

#filtered 10 country based GDP per capita

top_gdp_countries = df.nlargest(10, 'GDP per capita')

# 2. Line Chart: GDP per capita for top 10 countries
st.write('### GDP per capita by Top 10 Countries')
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x="Country or region", y="GDP per capita", data=top_gdp_countries, marker="o", ax=ax)
ax.set_title("GDP per capita by Top 10 Countries")
ax.set_xlabel("Country")
ax.set_ylabel("GDP per capita")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
st.pyplot(fig)



# Checkbox for Top 10 Countries by Generosity
if st.checkbox("Top 10 Countries by Generosity"):
    filtered_df = df.nlargest(10, "Generosity")
else:
    filtered_df = df

st.write("### Generosity by Country")
fig, ax = plt.subplots(figsize=(10, 12))
sns.barplot(y="Country or region", x="Generosity", data=filtered_df, ax=ax)  
ax.set_title("Generosity by Country")
ax.set_xlabel("Generosity")
ax.set_ylabel("Country")
st.pyplot(fig)


# 3. # Map: Generosity by Country
st.write("### Generosity by Country")
fig = px.choropleth(df, 
                    locations="Country or region", 
                    locationmode="country names", 
                    color="Generosity",
                    hover_name="Country or region",
                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(title="Generosity by Country", 
                  geo=dict(showframe=False, 
                           projection_type='natural earth'))

st.plotly_chart(fig)


# 4. Heatmap: Correlation between different indicators
st.write("### Correlation between Indicators")
fig, ax = plt.subplots(figsize=(10, 8))
correlation = df[["Score", "GDP per capita", "Social support", "Healthy life expectancy", 
                  "Freedom to make life choices", "Generosity", "Perceptions of corruption"]].corr()
sns.heatmap(correlation, annot=True, cmap="YlGnBu", ax=ax)
ax.set_title("Correlation between Indicators")
st.pyplot(fig)

# 5. Stacked Bar Plot: Freedom and Corruption by Top 10 Countries
top_countries = df.nlargest(10, 'Freedom to make life choices')
st.write("### Freedom and Corruption by Top 10 Countries (Filtered)")
fig, ax = plt.subplots(figsize=(12, 8))
top_countries.set_index("Country or region")[["Freedom to make life choices", "Perceptions of corruption"]].plot(
    kind="bar", stacked=True, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
ax.set_title("Freedom and Corruption by Country (Top 10)")
ax.set_xlabel("Country")
ax.set_ylabel("Values")
st.pyplot(fig)

# 6. Slider for filtering by Healthy life expectancy
min_life_exp, max_life_exp = st.slider("Select range for Healthy life expectancy", 
                                       float(df["Healthy life expectancy"].min()), 
                                       float(df["Healthy life expectancy"].max()), 
                                       (0.8, 1.0))
filtered_data = df[(df["Healthy life expectancy"] >= min_life_exp) & 
                   (df["Healthy life expectancy"] <= max_life_exp)]

st.write("### Healthy life expectancy by Country (Filtered)")
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x="Country or region", y="Healthy life expectancy", data=filtered_data, marker="o", ax=ax)
ax.set_title("Healthy life expectancy by Country (Filtered)")
ax.set_xlabel("Country")
ax.set_ylabel("Healthy life expectancy")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
st.pyplot(fig)

# 7. Checkbox for displaying Generosity by Country
if st.checkbox("Generosity by Country"):
    st.write("### Generosity by Country")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x="Country or region", y="Generosity", data=df, ax=ax)
    ax.set_title("Generosity by Country")
    ax.set_xlabel("Country")
    ax.set_ylabel("Generosity")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    st.pyplot(fig)

# 8. Dropdown for selecting a specific country and displaying its data
country = st.selectbox("Select a country", df["Country or region"].unique())
country_data = df[df["Country or region"] == country]
st.write(f"### Data for {country}")
st.write(country_data)


# 9. Correlation digram
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GDP per capita', y='Score', data=df)

plt.title('Relation between GDP per capita and Happiness Score', fontsize=16)
plt.xlabel('GDP per capita', fontsize=12)
plt.ylabel('Happiness Score', fontsize=12)
plt.show()

correlation = df['GDP per capita'].corr(df['Score'])
print(f"Correlation between GDP per capita and Happiness Score: {correlation:.2f}")


# 9.  Freedom and Corruption by Top 10 Countries
st.write("### Freedom and Corruption by Top 10 Countries")
fig, ax = plt.subplots(figsize=(12, 8))
x = range(len(top_countries))
width = 0.4

ax.bar(x, top_countries["Freedom to make life choices"], width=width, label="Freedom to Make Life Choices", color="blue", align='center')

ax.bar([p + width for p in x], top_countries["Perceptions of corruption"], width=width, label="Perceptions of Corruption", color="orange", align='center')

ax.set_xticks([p + width/2 for p in x])
ax.set_xticklabels(top_countries["Country or region"], rotation=45, ha="right", fontsize=10)
ax.set_title("Freedom and Corruption by Top 10 Countries", fontsize=16)
ax.set_xlabel("Country", fontsize=14)
ax.set_ylabel("Values", fontsize=14)
ax.legend()
st.pyplot(fig)    

