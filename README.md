#  EV-python-project
Just completed an EDA project on electric vehicles using Python! 📊 Explored trends, cleaned real-world data, and visualized key insights.
# 1. Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
plt.style.use('seaborn-v0_8')  # Using the correct style name
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 100

# Custom color palette
custom_palette = sns.color_palette("husl", 8)

# 2. Loading the Dataset
df = pd.read_csv("project.csv")

# 3. Data Cleaning
df.dropna(inplace=True)
df.drop(columns=["State"], inplace=True)
df.drop_duplicates(inplace=True)

# 4. EV Growth Over Time
ev_growth = df.groupby("Model Year").size().reset_index(name="Number of EVs")
plt.figure(figsize=(14, 7))
sns.lineplot(data=ev_growth, x="Model Year", y="Number of EVs", 
             marker="o", linewidth=2.5, color=custom_palette[0])
plt.title("Growth of Electric Vehicles Over Time", fontsize=18, pad=20)
plt.xlabel("Model Year", fontsize=14)
plt.ylabel("Number of EVs Registered", fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Top 10 EV Manufacturers
top_makes = df['Make'].value_counts().head(10)
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=top_makes.values, y=top_makes.index, palette=custom_palette)
plt.title("Top 10 EV Manufacturers", fontsize=18, pad=20)
plt.xlabel("Number of Vehicles", fontsize=14)
plt.ylabel("Make", fontsize=14)
for i, v in enumerate(top_makes.values):
    ax.text(v + 0.1, i, str(v), color='black', fontweight='bold')
plt.tight_layout()
plt.show()

# 6. Top 10 EV Models
top_models = df['Model'].value_counts().head(10)
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=top_models.values, y=top_models.index, palette=custom_palette)
plt.title("Top 10 EV Models", fontsize=18, pad=20)
plt.xlabel("Number of Vehicles", fontsize=14)
plt.ylabel("Model", fontsize=14)
for i, v in enumerate(top_models.values):
    ax.text(v + 0.1, i, str(v), color='black', fontweight='bold')
plt.tight_layout()
plt.show()

# 7. Distribution of EV Types
type_counts = df['Electric Vehicle Type'].value_counts()
plt.figure(figsize=(10, 8))
colors = sns.color_palette("Set2", len(type_counts))
labels = [f"{label}\n({count:,} - {count/sum(type_counts)*100:.1f}%)" for label, count in type_counts.items()]
plt.pie(type_counts, labels=labels, startangle=140, colors=colors, 
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}, 
        textprops={'fontsize': 12})
plt.title("Distribution of Electric Vehicle Types", fontsize=18, pad=20)
plt.axis('equal')
plt.tight_layout()
plt.show()

# 8. Top Cities by EV Count
top_cities = df['City'].value_counts().head(10)
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=top_cities.values, y=top_cities.index, palette=custom_palette)
plt.title("Top 10 Cities by EV Count", fontsize=18, pad=20)
plt.xlabel("Number of EVs", fontsize=14)
plt.ylabel("City", fontsize=14)
for i, v in enumerate(top_cities.values):
    ax.text(v + 0.1, i, str(v), color='black', fontweight='bold')
plt.tight_layout()
plt.show()

# 9. Year-over-Year Growth Rate
ev_growth["Yearly Growth Rate (%)"] = ev_growth["Number of EVs"].pct_change() * 100
plt.figure(figsize=(12, 8))
ax = sns.barplot(data=ev_growth[1:], x="Model Year", y="Yearly Growth Rate (%)", 
                palette=custom_palette)
plt.title("Year-over-Year EV Growth Rate", fontsize=18, pad=20)
plt.xlabel("Model Year", fontsize=14)
plt.ylabel("Growth Rate (%)", fontsize=14)
plt.xticks(rotation=45)
for i, v in enumerate(ev_growth[1:]["Yearly Growth Rate (%)"]):
    ax.text(i, v + 0.1, f"{v:.1f}%", ha='center', color='black', fontweight='bold')
plt.tight_layout()
plt.show()

# 10. Top EV Models by Type
top_models_by_type = df.groupby(['Electric Vehicle Type', 'Model']).size().reset_index(name='Count')
top_models_by_type = top_models_by_type.sort_values('Count', ascending=False).groupby('Electric Vehicle Type').head(5)

plt.figure(figsize=(14, 8))
sns.barplot(data=top_models_by_type, y='Model', x='Count', 
           hue='Electric Vehicle Type', palette=custom_palette)
plt.title("Top 5 EV Models by Type (BEV vs PHEV)", fontsize=18, pad=20)
plt.xlabel("Number of EVs", fontsize=14)
plt.ylabel("Model", fontsize=14)
plt.legend(title='Vehicle Type', title_fontsize=12, fontsize=12)
plt.tight_layout()
plt.show()

# 11. Correlation Heatmap
numerical_cols = ['Model Year', 'Electric Range', 'Base MSRP']
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', 
           fmt=".2f", linewidths=0.5, square=True)
plt.title("Correlation Heatmap", fontsize=18, pad=20)
plt.tight_layout()
plt.show()

# 12. Average Electric Range Over Time
avg_range_by_year = df.groupby("Model Year")["Electric Range"].mean().reset_index()

plt.figure(figsize=(12, 8))
sns.lineplot(data=avg_range_by_year, x="Model Year", y="Electric Range", 
            marker="o", linewidth=2.5, color=custom_palette[2])
plt.title("Average Electric Range Over Time", fontsize=18, pad=20)
plt.xlabel("Model Year", fontsize=14)
plt.ylabel("Average Electric Range (miles)", fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 13. Average EVs per City Over Time
avg_city_count = df.groupby(['Model Year', 'City']).size().groupby('Model Year').mean().reset_index(name='Average EVs per City')

plt.figure(figsize=(12, 8))
sns.lineplot(data=avg_city_count, x="Model Year", y="Average EVs per City", 
            marker="o", color=custom_palette[3], linewidth=2.5)
plt.title("Average EVs per City Over Time", fontsize=18, pad=20)
plt.xlabel("Model Year", fontsize=14)
plt.ylabel("Average Count", fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 14. Average MSRP Over Time
avg_msrp_by_year = df.groupby("Model Year")["Base MSRP"].mean().reset_index()

plt.figure(figsize=(12, 8))
sns.lineplot(data=avg_msrp_by_year, x="Model Year", y="Base MSRP", 
            marker="o", color=custom_palette[4], linewidth=2.5)
plt.title("Average MSRP Over Time", fontsize=18, pad=20)
plt.xlabel("Model Year", fontsize=14)
plt.ylabel("Average MSRP (USD)", fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 15. MSRP Distribution by Electric Vehicle Type
msrp_trend = df.groupby("Model Year")["Base MSRP"].mean().reset_index()
plt.figure(figsize=(12, 8))
sns.lineplot(data=msrp_trend, x="Model Year", y="Base MSRP", 
            marker="o", color=custom_palette[5], linewidth=2.5)
plt.title("Average MSRP of EVs Over Time", fontsize=18, pad=20)
plt.xlabel("Model Year", fontsize=14)
plt.ylabel("Base MSRP (USD)", fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



