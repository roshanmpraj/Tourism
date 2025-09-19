# for data manipulation
import pandas as pd
import sklearn
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths


os.environ["HF_TOKEN"] = "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"  # please use your token
api = HfApi(token=os.getenv("HF_TOKEN"))

# please create your dataset as you create your space
DATASET_PATH = "hf://datasets/Roshanmpraj/Tourism/tourism.csv"


df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# =============================
# Drop unique identifiers
# =============================
for col in ["CustomerID", "Unnamed: 0"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# =============================
# Encode categorical columns
# =============================
categorical_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# =============================
# Define target + features
# =============================
target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename

        repo_id="Roshanmpraj/Tourism",

        repo_type="dataset",
    )
