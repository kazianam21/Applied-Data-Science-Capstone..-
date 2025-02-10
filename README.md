# Applied-Data-Science-Capstone..-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("original_data.csv")

# Explore missing values
print(df.isnull().sum())
sns.heatmap(df.isnull(), cbar=False)
plt.show()

# Handle missing values (example: imputation with mean)
df['column_with_missing'].fillna(df['column_with_missing'].mean(), inplace=True)

# ... other cleaning steps ...

df.to_csv("cleaned_data.csv", index=False)
