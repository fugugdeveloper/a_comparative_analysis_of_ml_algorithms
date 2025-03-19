import pandas as pd

# Load the datasets
df_opencv = pd.read_csv('opencv_opencv_prs_before_update.csv')
df_spring = pd.read_csv('spring-projects_spring-framework_prs_before_update.csv')
df_django = pd.read_csv('django_django_prs_before_update.csv')
df_facebook = pd.read_csv('facebook_react_prs_before_update.csv')
df_codeigniter = pd.read_csv('codeigniter4_CodeIgniter4_prs_before_update.csv')
df_rails = pd.read_csv('rails_rails_prs_before_update.csv')

df_opencv_new = pd.read_csv('opencv_opencv_prs.csv')
df_spring_new = pd.read_csv('spring-projects_spring-framework_prs.csv')
df_django_new = pd.read_csv('django_django_prs.csv')
df_facebook_new = pd.read_csv('facebook_react_prs.csv')
df_codeigniter_new = pd.read_csv('codeigniter4_CodeIgniter4_prs.csv')
df_rails_new = pd.read_csv('rails_rails_prs.csv')
df_kotlin_new = pd.read_csv('android_compose-samples_prs.csv')

# Concatenate the datasets
merged_df = pd.concat([df_opencv, df_spring, df_django, df_facebook, df_rails, df_codeigniter,
                       df_opencv_new,
                       df_spring_new, df_django_new, df_facebook_new, df_codeigniter_new, df_rails_new, df_kotlin_new],
                      ignore_index=True)

# Save the merged dataset to a new CSV file (optional)
merged_df.to_csv('all_merged_prs.csv', index=False)

print(merged_df.head())
print('success')