<img width="2578" height="1395" alt="image" src="https://github.com/user-attachments/assets/c536ba0f-6d64-4cfb-9e63-576b7ac425f5" />
<img width="2578" height="387" alt="image" src="https://github.com/user-attachments/assets/525a96d3-72cc-4c5f-891f-c6bb117a01f4" />

## Getting Started

### 1. Set Up the Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. AWS Prerequisites

Before running any notebooks, make sure you've completed the following steps:

1. **Install boto3 and AWS CLI** — both are listed in `requirements.txt` and will be installed automatically.
2. **Create an IAM User** — go to the AWS Console and create a user with **Administrator Access**, or at minimum grant permissions for EC2 and S3.
3. **Generate Access Keys** — in the IAM user settings, create an **Access Key** for CLI usage. You'll get an *Access Key ID* and a *Secret Access Key* — keep these safe.
4. **Store credentials as environment variables** — never hardcode them in your code.
5. **Configure the AWS CLI** — run `aws configure` and provide your Access Key ID, Secret Access Key, and default region. This writes the credentials to `~/.aws/credentials`.

> **Note:** The `.pem` file is used for SSH access to EC2 instances. Your Access Key and Secret Key are what authenticate API calls for creating instances, S3 buckets, etc.

---

### 3. ML Model

- Trains a **Random Forest** classification model on the Loan Dataset (`Dataset/Loan_Dataset.csv`).
- The trained model (`random_forest_model.pkl`) and scaler (`scaler.pkl`) are saved to the `models/` folder and uploaded to the S3 bucket (`loan-dataset-models`).
- See `ml_model_training/code.ipynb` to retrain the model.

---

### 4. Running the Streamlit App

```bash
streamlit run app.py
```

**App URL:** http://localhost:8501

**On startup**, the app automatically:
1. Checks if `models/random_forest_model.pkl` and `models/scaler.pkl` exist locally.
2. If either file is missing, downloads it from the S3 bucket.
3. Loads both files into memory — ready for predictions.

> Models are cached with `@st.cache_resource`, so S3 is only contacted once per server session.


