# for data manipulation
import pandas as pd
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = "ProdTaken"

# List of numerical features in the dataset
numeric_features = [
    'Age',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'PitchSatisfactionScore',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Passport',
    'OwnCar',
    'Designation',
]

df["Gender"] = df["Gender"].replace("Fe Male", "Female")

df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

# Define predictor matrix (X) using selected numeric and categorical features
X = df[numeric_features + categorical_features]

# Define target variable
y = df[target]

# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42,   # Ensures reproducibility by setting a fixed random seed
    stratify=y,        # keeps the (imbalanced) churn ratio consistent across splits
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
