import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load your data
data = pd.read_csv('C:/Users/unknown user/Desktop/DEP Task 1/Daily Household Transactions.csv')

# Check the column names
print("Column names:", data.columns)

# Print the first few rows to inspect data
print(data.head())

# Handle missing values
data = data.dropna(subset=['Amount', 'Income/Expense'])  # Drop rows with missing 'Amount' or 'Income/Expense'

# Filter for 'Expense' records
data = data[data['Income/Expense'] == 'Expense']

# Group by 'Mode' (assuming 'Mode' is used as a proxy for household in the absence of 'HouseholdID')
household_data = data.groupby('Mode').agg({
    'Amount': 'sum',
    'Mode': 'count'
}).rename(columns={'Amount': 'Total_Spending', 'Mode': 'Num_Transactions'})

# Add dummy 'Household_Lifetime' (you can adjust this based on your needs or use a default value)
household_data['Household_Lifetime'] = 0  # Replace with a meaningful default value if possible

# Add a placeholder for average spending per transaction
household_data['Avg_Spending_Per_Transaction'] = household_data['Total_Spending'] / household_data['Num_Transactions']

# EDA
print(household_data.describe())
sns.histplot(household_data['Total_Spending'], bins=50)
plt.show()
sns.boxplot(x=household_data['Num_Transactions'])
plt.show()
corr_matrix = household_data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(household_data[['Total_Spending', 'Num_Transactions', 'Avg_Spending_Per_Transaction']])

# Determine the number of clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Fit the K-means model
optimal_clusters = 4  # Example: choose 4 clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
household_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Silhouette score
sil_score = silhouette_score(scaled_data, household_data['Cluster'])
print(f'Silhouette Score: {sil_score}')

# Visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=household_data['Cluster'], palette='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Household Segments')
plt.show()

# Profile clusters
cluster_profile = household_data.groupby('Cluster').mean()
print(cluster_profile)


