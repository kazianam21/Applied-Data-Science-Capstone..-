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


import pandas as pd

df = pd.read_csv("cleaned_data.csv")

# Example: Creating a new feature (ratio)
df['new_feature'] = df['column1'] / df['column2']

# Example: One-hot encoding categorical variables
df = pd.get_dummies(df, columns=['categorical_column'])

df.to_csv("engineered_data.csv", index=False)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("engineered_data.csv")

X = df.drop('target_variable', axis=1)  # Features
y = df['target_variable']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# ... model comparison, hyperparameter tuning ...
