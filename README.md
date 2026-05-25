# 🏦 Loan Approval Predictor — Streamlit + AWS

<img width="2840" height="1467" alt="image" src="https://github.com/user-attachments/assets/99a8eb06-4278-484f-8a50-40e328fe6901" />
<img width="2834" height="1465" alt="image" src="https://github.com/user-attachments/assets/3074ab28-20ce-45e8-9907-3058dc09a897" />
<img width="2810" height="1476" alt="image" src="https://github.com/user-attachments/assets/e6d5dad6-e96b-46e9-9934-380b1c6f2f23" />

A **Streamlit** web app for **loan approval prediction** (binary classification) powered by a **Random Forest** model, with end-to-end AWS integration.

**What it does**
- Upload or fetch the dataset and trained model from **Amazon S3**.
- Enter applicant details in the UI and get an instant **Approved / Rejected** prediction with confidence.
- Models are downloaded on demand via a sidebar button — predictions are blocked until artifacts are present locally.

**AWS stack**
- **S3** — stores the dataset (`Dataset/`) and trained artifacts (`models/random_forest_model.pkl`, `scaler.pkl`) in the `loan-dataset-models` bucket.
- **IAM** — dedicated user with scoped permissions (`s3:CreateBucket`, `s3:PutObject`, `s3:GetObject`, `s3:ListBucket`) for secure access; credentials loaded from `.env` via `python-dotenv`.
- **EC2** — Amazon Linux instance hosts the Streamlit app, pulls the latest model from S3 at runtime, and serves the UI on port 8501.

**ML**
- Task: **binary classification** (Loan_Status: Approved / Rejected).
- Algorithm: `RandomForestClassifier` (scikit-learn), with `StandardScaler` on 11 engineered features.

---



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


