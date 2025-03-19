import pandas as pd
from sklearn.preprocessing import RobustScaler

# Load the dataset
file_path = 'django_django_prs.csv'
df = pd.read_csv(file_path)
df = df.drop(columns=['pr_id'])
# Identify and drop columns with all NaN values
df = df.dropna(axis=1, how='all')

# Separate numeric and non-numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns

# Fill NaN values in numeric columns with the median
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Initialize the RobustScaler
scaler = RobustScaler()

# Fit and transform the numeric data
df_normalized = pd.DataFrame(scaler.fit_transform(df[numeric_columns]), columns=numeric_columns)

# Concatenate non-numeric columns back to the normalized data
df_normalized = pd.concat([df[non_numeric_columns], df_normalized], axis=1)

# Save the normalized dataset if needed
df_normalized.to_csv('robust_scaled_django_django_prs.csv', index=False)

# Display the first few rows of the normalized dataset
print(df_normalized.head())
