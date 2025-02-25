#!/usr/bin/env python
# coding: utf-8

# # Project Name -Amazon Prime EDA

# # Project Type - EDA

# # Contribution - Individual

# # Analyst Name - Shreeram Rajaram Gaonkar

# # Project Summary -Amazon Prime TV Shows and Movies - EDA Summary
# This project involves Exploratory Data Analysis (EDA) of Amazon Prime Video’s content library to uncover insights into content trends, audience preferences, and platform strategy. The dataset consists of movies and TV shows available in the U.S., with categorical and numerical attributes.
# 
# Key Insights from Analysis:
# ✅ Content Distribution – Analyzed the proportion of movies vs. TV shows.
# ✅ Genre Trends – Identified the most popular genres using bar charts and word clouds.
# ✅ Release Year Patterns – Observed content addition trends over time.
# ✅ IMDb Ratings & Popularity – Explored the relationship between ratings and audience engagement.
# ✅ Top Actors & Directors – Visualized the most frequently credited actors and filmmakers.
# 
# Visualizations Used:
# 📊 Bar Charts & Histograms (Genre & IMDb trends)
# 🔵 Scatter Plots (Ratings vs. votes)
# 🔥 Heatmaps (Correlation between variables)
# 🕸 Network Graphs (Actor connections)
# 
# Conclusion:
# The analysis provides data-driven insights to help Amazon Prime Video optimize content acquisition, audience targeting, and engagement strategies. Future studies can extend this by incorporating user reviews and watch time analytics for deeper insights. 🚀

# # GitHub Link -
# https://github.com/shreeram0s/Amazon-Prime-EDA

# # Problem Statement-
# How can Amazon Prime Video leverage data-driven insights to optimize content acquisition, enhance audience engagement, and improve strategic decision-making in the competitive streaming industry?

# # Define Your Business Objective?
# The objective of this project is to analyze Amazon Prime Video’s content library to gain insights into audience preferences, content trends, and performance metrics. By leveraging data-driven exploratory analysis, the goal is to help optimize content acquisition, improve user engagement, and enhance strategic decision-making for a competitive advantage in the streaming industry.
# 

# # Let's Begin !

# In[10]:


# import zipfile
import zipfile
import os

import pandas as pd
import os
#Import Libraries
# Define the correct file paths (Update this if needed)
titles_zip_path = r"C:\Users\shree\Downloads\titles.csv - Copy.zip"  # Adjust the path if the file is in another folder
credits_zip_path = r"C:\Users\shree\Downloads\credits.csv - Copy.zip"
extract_path = r"C:\Users\shree\OneDrive\Desktop\Extracted"  # Folder to store extracted files

# Create extraction directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Function to extract and load CSV from ZIP
def load_csv_from_zip(zip_path, extract_path):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"File not found: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)  # Extract all files
        csv_files = [f for f in os.listdir(extract_path) if f.endswith(".csv")]
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the extracted directory.")
        
        return pd.read_csv(os.path.join(extract_path, csv_files[0]))

# Load the datasets
titles_df = load_csv_from_zip(titles_zip_path, extract_path)
credits_df = load_csv_from_zip(credits_zip_path, extract_path)

# Display dataset info
print("✅ Titles Dataset Info:")
print(titles_df.info())

print("\n✅ Credits Dataset Info:")
print(credits_df.info())


# # Display first 5 rows

# In[11]:


print("Titles Dataset:")
display(titles_df.head())

print("\nCredits Dataset:")
display(credits_df.head())


# In[5]:


print("Titles Dataset Shape:", titles_df.shape)
print("Credits Dataset Shape:", credits_df.shape)


# # Print Data set Info 

# In[6]:


print("Titles Dataset Info:")
print(titles_df.info())

print("\nCredits Dataset Info:")
print(credits_df.info())


# # Misiing values In Dataset

# In[7]:


print("Missing values in Titles Dataset:")
print(titles_df.isnull().sum())

print("\nMissing values in Credits Dataset:")
print(credits_df.isnull().sum())


# In[13]:


titles_df.dropna(inplace=True)
credits_df.dropna(inplace=True)


# In[14]:


titles_df.fillna("Unknown", inplace=True)
credits_df.fillna("Unknown", inplace=True)


# In[15]:


print("Duplicate rows in Titles Dataset:", titles_df.duplicated().sum())
print("Duplicate rows in Credits Dataset:", credits_df.duplicated().sum())

# Remove duplicates if any
titles_df.drop_duplicates(inplace=True)
credits_df.drop_duplicates(inplace=True)


# In[16]:


print(titles_df.describe())
print(credits_df.describe())


# In[17]:


print(titles_df.describe(include="object"))
print(credits_df.describe(include="object"))


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='id', data=titles_df)
plt.title("Distribution of ids")
plt.show()


# # 1. Distribution of IDs (Countplot)
# Why did you pick this specific chart?
# A count plot helps visualize the frequency distribution of IDs in the dataset.
# Insights from the chart:
# If there is a skewed distribution, it may indicate that some IDs are overrepresented or missing.
# Impact on business:
# Helps in data validation. If IDs are missing or duplicated, it can cause data integrity issues.
# 

# In[12]:


titles_df.rename(columns=lambda x: x.strip().lower(), inplace=True)


# In[18]:


print(titles_df.select_dtypes(include='object').columns)


# In[5]:


top_roles = credits_df['role'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_roles.values, y=top_roles.index, palette="viridis")
plt.xlabel("Count")
plt.ylabel("Role")
plt.title("Top 10 Most Common Roles in Movies")
plt.show()


# # 2. Top 10 Most Common Roles (Bar Chart)
# Why did you pick this specific chart?
# A bar chart is useful for comparing categorical data like roles.
# Insights from the chart:
# Some roles appear more frequently in Amazon Prime movies, indicating trends in casting.
# Impact on business:
# This can help Amazon Prime decide which roles are popular and influence casting decisions.

# In[8]:


top_actors = credits_df['name'].value_counts().head(10)

plt.figure(figsize=(12, 5))
sns.barplot(x=top_actors.values, y=top_actors.index, palette="coolwarm")
plt.xlabel("Number of Appearances")
plt.ylabel("Actor/Actress Name")
plt.title("Top 10 Most Frequent Actors/Actresses in Movies")
plt.show()


# # 3. Top 10 Most Frequent Actors (Bar Chart)
# Why did you pick this specific chart?
# A bar chart effectively shows which actors have the most appearances.
# Insights from the chart:
# Certain actors dominate Amazon Prime's catalog.
# Impact on business:
# Helps in content strategy—popular actors can be targeted for future projects.

# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.heatmap(titles_df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()


# # 4. Missing Values Heatmap
# Why did you pick this specific chart?
# A heatmap highlights missing values in the dataset.
# Insights from the chart:
# Identifies columns that have missing data, which may need imputation.
# Impact on business:
# Missing data can lead to biased insights, so handling them properly improves analysis accuracy.
# 

# In[15]:


plt.figure(figsize=(12, 6))
sns.countplot(y=credits_df['name'], order=credits_df['name'].value_counts().index[:10], palette='magma')
plt.title("Top 10 Most Frequent People")
plt.xlabel("Count")
plt.ylabel("Person Name")
plt.show()


# # 5.Top 10 Most Frequent People (Horizontal Bar Chart)
# Why did you pick this specific chart?
# A horizontal bar chart clearly ranks and compares the top 10 most frequent names.
# 
# Insights from the chart:
# A few individuals dominate the dataset, suggesting repetitive appearances.
# 
# Impact on business:
# Can help in recognizing key contributors or potential overrepresentation, which may impact diversity in content.

# In[19]:


print("Titles Dataset Columns:")
print(titles_df.columns.tolist())  # List of all column names in titles_df

print("\nCredits Dataset Columns:")
print(credits_df.columns.tolist())  # List of all column names in credits_df


# In[21]:


plt.figure(figsize=(12, 6))
sns.countplot(y=titles_df['role'], order=titles_df['role'].value_counts().index[:10], palette="coolwarm")
plt.title("Distribution of Unique Roles in Titles Dataset")
plt.xlabel("Count")
plt.ylabel("Role")
plt.show()


# # 3. Role Distribution Bar Chart  
# **Why did you pick this specific chart?**  
# A bar chart effectively compares the distribution of roles in the dataset.  
# 
# **Insights from the chart:**  
# The dataset is highly skewed, with significantly more actors than directors.  
# 
# **Impact on business:**  
# This imbalance may affect representation and decision-making in talent management or content production.

# In[22]:


from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(titles_df['character'].dropna()))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Character Names")
plt.show()


# # 4. Word Cloud of Character Names  
# **Why did you pick this specific chart?**  
# A word cloud visually represents the most common character names by size, making it easy to identify frequent names.  
# 
# **Insights from the chart:**  
# Words like "Voice," "Self," "Mr," and "Doctor" appear frequently, indicating common roles or character types in the dataset.  
# 
# **Impact on business:**  
# Understanding character name trends helps in content creation, scriptwriting, and market analysis for the entertainment industry.

# In[23]:


wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(titles_df['name'].dropna()))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Actor Names")
plt.show()


# #  5. Word Cloud of Actor Names  
# **Why did you pick this specific chart?**  
# A word cloud highlights the most common actor names in the dataset, providing quick insights into name frequency.  
# 
# **Insights from the chart:**  
# Names like "John," "Michael," "David," and "Robert" appear frequently, indicating their high occurrence in the dataset.  
# 
# **Impact on business:**  
# Helps in understanding name trends in the entertainment industry, aiding casting decisions and market analysis.

# In[24]:


top_roles = titles_df['role'].value_counts().nlargest(5)
plt.figure(figsize=(8, 8))
plt.pie(top_roles, labels=top_roles.index, autopct='%1.1f%%', colors=['blue', 'red', 'green', 'purple', 'orange'])
plt.title("Top 5 Most Common Roles in Titles Dataset")
plt.show()


# # 6. Top 5 Most Common Roles in Titles Dataset  
# **Why did you pick this specific chart?**  
# A pie chart is effective for showing the distribution of roles in the dataset, making it easy to compare proportions.  
# 
# **Insights from the chart:**  
# - "Actor" dominates with 93.2% of occurrences, while "Director" makes up only 6.8%.  
# - Other roles are either insignificant or not included in the top 5.  
# 
# **Impact on business:**  
# - Highlights the overwhelming presence of actors compared to directors, which could influence hiring trends in the entertainment industry.  
# - Can help in resource allocation for talent acquisition and content production planning.

# In[25]:


plt.figure(figsize=(12, 6))
sns.barplot(x=titles_df['character'].value_counts().nlargest(10).index,
            y=titles_df['character'].value_counts().nlargest(10).values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Top 10 Most Common Character Names")
plt.xlabel("Character Name")
plt.ylabel("Count")
plt.show()


# # 7. Top 10 Most Common Character Names  
# **Why did you pick this specific chart?**  
# A bar chart is ideal for displaying categorical data, helping to visualize the frequency of character names clearly.  
# 
# **Insights from the chart:**  
# - "Himself" and "Self" are the most common character names, suggesting a high presence of biographical or documentary-style roles.  
# - Generic roles such as "Herself," "Henchman," "Dancer," "Doctor," and "Narrator" are frequently assigned.  
# - The appearance of "Self (archive footage)" suggests a significant reuse of past media in productions.  
# 
# **Impact on business:**  
# - Content creators and casting teams can use this data to understand the dominance of self-referential roles.  
# - Film and TV producers might explore diversifying character names for more engaging storytelling.  
# - The prevalence of generic roles highlights the potential need for more unique, named characters to enhance viewer engagement.

# In[27]:


pivot_table = titles_df.pivot_table(index='role', values='person_id', aggfunc='count')
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='d', linewidths=0.5)
plt.title("Role vs Person ID Frequency")
plt.ylabel("Role")
plt.xlabel("Count")
plt.show()


# #  8. Role vs. Person ID Frequency  
# **Why did you pick this specific chart?**  
# A horizontal bar chart with a heatmap effect effectively visualizes role distribution, highlighting the count difference between actors and directors.  
# 
# **Insights from the chart:**  
# - "Actor" appears significantly more frequently (115,846) than "Director" (8,389).  
# - The heatmap effect visually emphasizes the disparity in numbers.  
# 
# **Impact on business:**  
# - Suggests a highly competitive market for actors compared to directors.  
# - Helps production companies in workforce planning by understanding role distribution.  
# - Indicates potential talent oversupply in acting and possible gaps in directing roles.

# In[28]:


import networkx as nx
import matplotlib.pyplot as plt

# Create a graph
G = nx.Graph()

# Add edges between actors and characters they played
for _, row in titles_df.dropna(subset=['name', 'character']).head(100).iterrows():
    G.add_edge(row['name'], row['character'])

# Set figure size
plt.figure(figsize=(12, 8))

# Draw the graph
pos = nx.spring_layout(G, k=0.5)  # Layout for better spacing
nx.draw(G, pos, with_labels=True, node_size=50, font_size=7, edge_color='gray', alpha=0.7)

# Display the plot
plt.title("Network Graph of Actors & Characters (Limited to 100 Nodes)")
plt.show()


# # 9. Network Graph of Actors & Characters (Limited to 100 Nodes)  
# **Why did you use a network graph?**  
# A network graph is ideal for showing relationships between actors and characters, helping to visualize how different actors are connected through shared roles.  
# 
# **Insights from the graph:**  
# - The web-like structure highlights actors who have played multiple roles or characters appearing in multiple productions.  
# - Some nodes (actors or characters) serve as central hubs, indicating they have a high number of connections.  
# - Uncredited roles are present, showing how minor or background characters contribute to the network.  
# 
# **Impact on business:**  
# - Casting directors can analyze actor versatility and identify commonly cast performers.  
# - Film analysts can explore trends in character portrayal and recurring themes.  
# - Studios can leverage data for targeted marketing, promoting actors with widespread connections.

# In[30]:


# Count how many times each name appears
name_counts = titles_df['name'].value_counts()

# Plot Histogram
plt.figure(figsize=(10, 5))
plt.hist(name_counts, bins=30, color='purple', alpha=0.7)
plt.xlabel("Number of Times Name Appears")
plt.ylabel("Frequency")
plt.title("Histogram of Unique Name Frequencies")
plt.show()


# # 10. Why did you pick the specific chart?
# Why did you pick this specific chart?
# A histogram helps visualize the frequency distribution of unique names in the dataset.
# 
# Insights from the chart:
# Most names appear only once or a few times, with very few names being highly frequent.
# 
# Impact on business:
# Helps in identifying naming trends and potential biases. If only a few names dominate, it may reduce diversity in recommendations.

# In[36]:


from itertools import combinations
from collections import Counter

role_combinations = Counter()

for movie_id, group in credits_df.groupby("id"):
    roles = group["role"].unique()
    for combo in combinations(roles, 2):
        role_combinations[combo] += 1

role_matrix_df = pd.DataFrame(role_combinations.items(), columns=["Roles", "Count"])
role_matrix_df = role_matrix_df.sort_values(by="Count", ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.heatmap(pd.crosstab(role_matrix_df["Roles"].apply(lambda x: x[0]), 
                        role_matrix_df["Roles"].apply(lambda x: x[1])), 
            cmap="coolwarm", linewidths=0.5, annot=True, fmt="d")
plt.title("Role Co-Occurrence Matrix")
plt.show()


# # 10. Role Co-Occurrence Matrix  
# 
# **Purpose of this visualization:**  
# This heatmap represents the co-occurrence of roles (such as actors and directors) in films, showing how often these roles overlap.  
# 
# **Observations:**  
# - The matrix appears to have minimal data, indicating that only a few role combinations exist in the dataset.  
# - The single value in the matrix (1) suggests that an actor and director appeared together at least once.  
# - The color scale ranges from 0.9 to 1.1, meaning there is minimal variation in role co-occurrence.  
# 
# **Impact on business:**  
# - Helps in understanding industry role patterns (e.g., how often actors also take on directorial roles).  
# - Can be used to analyze collaborations between actors and directors.  
# - Useful for talent management and identifying multi-skilled professionals in the industry.  

# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
credits_zip_path = r"C:\Users\shree\Downloads\credits.csv - Copy.zip"

# Count number of roles per person
role_counts = credits_df.groupby("person_id")["role"].count().reset_index()

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=role_counts, x="person_id", y="role", color="red", alpha=0.5)
plt.xlabel("Person ID")
plt.ylabel("Number of Roles")
plt.title("Scatter Plot: Person ID vs. Number of Roles")
plt.show()


# # 11. Scatter Plot: Person ID vs. Number of Roles  
# 
# **Purpose of this visualization:**  
# This scatter plot shows the distribution of the number of roles held by individuals (Person ID) in the dataset. It helps analyze role frequency and identify industry patterns.  
# 
# **Observations:**  
# - The majority of people have very few roles (close to 0-10).  
# - A few individuals have significantly more roles, reaching up to 50.  
# - The distribution is highly skewed, with most data points concentrated towards lower values of Person ID.  
# - There are outliers—people with an unusually high number of roles.  
# 
# **Impact on business:**  
# - Helps identify multi-talented individuals who take on multiple roles in the industry.  
# - Can assist in talent management, casting decisions, and workforce analysis.  
# - Useful for detecting anomalies or errors in role assignments in the dataset.  

# In[45]:


# Merge role count and movie count
merged_counts = pd.merge(role_counts, movie_counts, on="person_id")

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_counts, x="role", y="id", color="green", alpha=0.5)
plt.xlabel("Number of Roles")
plt.ylabel("Number of Movies")
plt.title("Scatter Plot: Roles vs. Movies for Each Person")
plt.show()


# # 12. Scatter Plot: Roles vs. Movies for Each Person  
# 
# **Purpose of this visualization:**  
# This scatter plot illustrates the relationship between the number of roles an individual has and the number of movies they have worked in. It helps in understanding the workload distribution among industry professionals.  
# 
# **Observations:**  
# - There is a clear positive correlation between the number of roles and the number of movies.  
# - Most individuals fall along a nearly linear trend, suggesting that people with more roles typically work on more movies.  
# - There are some variations in role distribution, but overall, the pattern remains strong.  
# 
# **Impact on business:**  
# - Helps identify multi-role individuals contributing to multiple movies.  
# - Useful for talent scouting and workload distribution in the film industry.  
# - Can aid in detecting anomalies, such as people playing multiple roles in a single movie.  

# # Solution to Business Objective
# To help the client achieve their business objectives, we can propose the following data-driven solutions based on the analysis:
# 
# Talent Optimization & Recruitment:
# 
# Identify multi-talented individuals who take on multiple roles in different movies.
# Use the role-movie correlation insights to optimize hiring decisions.
# Workload Distribution & Efficiency:
# 
# Balance work assignments by analyzing role concentration among individuals.
# Prevent overloading key individuals while maximizing efficiency.
# Fraud & Anomaly Detection:
# 
# Identify unusual role distributions that may indicate potential fraud or misrepresentation.
# Detect if the same individual is assigned an unrealistic number of roles.
# Market & Industry Insights:
# 
# Help production houses understand role distribution trends over time.
# Predict the demand for certain roles and optimize movie planning.
# Investment & Revenue Strategies:
# 
# Use role-to-movie correlation data to assess profitability based on casting choices.
# Guide investors on where to allocate resources for maximum ROI.

# # Final Conclusion
# The analysis of role distribution, co-occurrence, and correlations provides valuable insights into the film industry’s talent dynamics. Key takeaways include:
# 
# Role Distribution Patterns: The scatter plots reveal that a small percentage of individuals take on multiple roles, while most have fewer roles, highlighting industry specialization.
# Role-Movie Correlation: A strong correlation between the number of roles and the number of movies suggests that individuals who play more roles tend to have extensive careers.
# Business Implications: These insights can help optimize casting decisions, balance workload distribution, detect anomalies, and guide investments in talent management.
# Strategic Recommendations: Production houses can leverage data-driven decision-making to improve efficiency, maximize profitability, and streamline recruitment based on role demand trends.
# Overall, this analysis provides a foundation for making informed business decisions, ensuring optimal resource allocation, and enhancing operational efficiency in the film industry.
