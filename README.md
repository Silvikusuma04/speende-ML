# Spendee - Machine Learning Repository

This repository contains all the Machine Learning models that form the core of the **Spendee** web application.  
Spendee is an application designed to support investors and financial institutions in assessing the viability of startup funding and individual loans.  
This project aims to minimize the risk of funding errors by providing a transparent and efficient data-driven platform.

## Machine Learning Model Descriptions

The Spendee application is powered by three main machine learning models developed to provide accurate predictions.

### 1. Startup Success Prediction
This model predicts the success potential of a startup based on various business and market parameters.

* **Objective**: To provide investors with an objective prediction of a startup's success potential, aiding in more informed investment decisions.
* **Architecture**: Built using a TensorFlow-based Neural Network architecture.
* **Accuracy**: 83%. 
* **Input Features Used**:
    * Initial Achievement Date
    * Last Achievement Date
    * First Funding Date
    * Last Funding Date
    * Number of Relations or Investors
    * Total Funding (in IDR)
    * Average Participants or Customers
    * Number of Funding Rounds
    * Number of Achievements
    * Funding per Relation Ratio (in IDR)
    * Average Funding per Round (in IDR)
    * Category (e.g., E-Commerce, Finance, Health, etc.)
    * Is Popular (Yes/No)

### 2. Loan Approval Prediction
This model predicts the likelihood of an individual's loan application being approved.

* **Objective**: To help banks and financial institutions accelerate the loan approval process and reduce the risk of credit default.
* **Architecture**: Developed using a TensorFlow Neural Network.
* **Accuracy**: 95%.
* **Input Features Used**:
    * Debt to Income Ratio
    * Monthly & Annual Income
    * Applied & Initial Interest Rate
    * Loan Amount & Monthly Loan Payment
    * Education Level
    * Net Worth & Total Assets
    * Applicant Age & Number of Dependents
    * Credit Score
    * Work Experience (Years)
    * Credit History Duration & Active Credit Lines
    * Loan Payment Period (Months)
    * Monthly Debt Payment
    * Savings Balance
    * Number of Credit Inquiries

## Tech Stack
* **Programming Language**: Python
* **API Framework**: Flask
* **Machine Learning Libraries**: TensorFlow, Google Cloud Vertex AI
* **Deployment Platform**: Google Cloud Run
* **CI/CD**: GitHub Actions

## API Endpoints & Usage

The predictive models have been deployed and are accessible via HTTP API endpoints on Google Cloud Platform.

### 1. API - Startup Success Prediction
* **URL**: `https://speende-1-ml-325126223708.europe-west1.run.app/predict`
* **Method**: `POST`
* **Example `curl` request**:
    ```bash
    curl -X POST https://speende-1-ml-325126223708.europe-west1.run.app/predict \
    -H "Content-Type: application/json" \
    -d '{
          "initial_achievement_date": "2020-01-01",
          "last_achievement_date": "2024-06-01",
          "first_funding_date": "2021-03-15",
          "last_funding_date": "2023-11-20",
          "investor_relations_count": 8,
          "total_funding_rp": 5000000000,
          "avg_participants_customers": 15000,
          "funding_rounds": 4,
          "achievements_count": 12,
          "funding_per_relation_ratio_rp": 625000000,
          "avg_funding_per_round_rp": 1250000000,
          "category": "E-Commerce",
          "is_popular": true
        }'
    ```

### 2. API - Loan Approval Prediction
* **URL**: `https://speende-fitur2-325126223708.us-central1.run.app/predict`
* **Method**: `POST`
* **Example `curl` request**:
    ```bash
    curl -X POST https://speende-fitur2-325126223708.us-central1.run.app/predict \
    -H "Content-Type: application/json" \
    -d '{
          "debt_to_income_ratio": 0.35,
          "monthly_income": 15000000,
          "annual_income": 180000000,
          "interest_rate": 12.5,
          "loan_amount": 100000000,
          "initial_interest_rate": 5.0,
          "education_level": "Bachelor",
          "net_worth": 250000000,
          "monthly_loan_payment": 5000000,
          "total_assets": 400000000,
          "applicant_age": 35,
          "credit_score": 750,
          "work_experience_years": 10,
          "credit_history_duration_years": 8,
          "loan_payment_period_months": 24,
          "monthly_debt_payment": 3000000,
          "savings_balance": 50000000,
          "credit_inquiries_count": 2,
          "dependents_count": 2,
          "active_credit_lines": 4
        }'
    ```

## Running the Project Locally

1.  **Clone this repository:**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd <PROJECT_FOLDER_NAME>
    ```

2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install all required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask application:**
    ```bash
    flask run
    ```
    The application will run at `http://127.0.0.1:5000`.
