import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'robust_scaled_django_django_prs.csv'
df = pd.read_csv(file_path)

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Initialize the LabelEncoder
le = LabelEncoder()

# Apply Label Encoding to each categorical column
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Now all categorical features have been converted to numeric values
df.to_csv('labelled_robust_scaled_django_django_prs.csv', index=False)
print(df.head())

# Proceed with model training, testing, etc.
