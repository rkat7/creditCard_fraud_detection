# Loading the dataset from the csv file using pandas
data = pd.read_csv('creditcard.csv')

# Exploring the dataset
print(data.columns)

# Print the shape of the data
data = data.sample(frac=0.1, random_state = 1)
print(data.shape)
print(data.describe())

# V1 - V28 are the results of a PCA Dimensionality reduction to protect user identities and sensitive features

# Plot histograms of each parameter 
data.hist(figsize = (20, 20))
plt.show()



