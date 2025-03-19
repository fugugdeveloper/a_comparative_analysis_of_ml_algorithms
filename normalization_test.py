import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import re

# Load data
file_path = 'all_merged_prs.csv'
df = pd.read_csv(file_path)



# Drop columns with no data and original 'security_issues'
df_cleaned = df.drop(columns=['author_contributions', 'pr_id', 'security_issues'])

# Fill missing values for numerical columns using median
num_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
simpleImputer = SimpleImputer(strategy='median')
df_cleaned[num_cols] = simpleImputer.fit_transform(df_cleaned[num_cols])

# Encode categorical columns
cat_cols = df_cleaned.select_dtypes(include=['object', 'bool']).columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))
    label_encoders[col] = le

# Normalize numerical features
scaler = StandardScaler()
df_cleaned[num_cols] = scaler.fit_transform(df_cleaned[num_cols])

# Save the cleaned and normalized dataset
df_cleaned.to_csv('merged_normalized_dataset_prs.csv', index=False)

# Display the first few rows
print(df_cleaned.head())