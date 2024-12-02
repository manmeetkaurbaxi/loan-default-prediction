# Overview
Loan default prediction is a critical application in the financial industry, where lenders and institutions aim to assess the risk associated with providing loans to individuals or businesses. The goal is to identify potential borrowers more likely to default on their loans, allowing lenders to make informed decisions and mitigate financial risks.

In the context of loan default prediction, machine learning models have shown promising results in accurately predicting whether a borrower will default on a loan. These models leverage historical data, such as past loan performance, financial history, employment details, and other relevant factors, to predict the likelihood of future loan defaults.

However, as machine learning models become increasingly complex, a significant challenge arises the lack of transparency. Many modern machine learning algorithms, such as deep neural networks, can be seen as "black boxes." While they may produce accurate predictions, how they arrive at those predictions is often unclear. This opacity raises concerns for various stakeholders involved in the loan lending process:

Data Scientists: Data scientists who develop and deploy these models need to understand their inner workings for debugging, improvement, and model selection. They require insights into feature importance and decision processes to ensure models are reliable and accurate.

Stakeholders: Business stakeholders, such as loan officers and management teams, need to trust and comprehend the model's predictions to make well-informed lending decisions. Clear explanations are essential for building trust and facilitating collaboration between data scientists and business units.

Regulators: Regulatory bodies and compliance officers demand transparency in lending practices. They require models to be explainable and auditable to ensure fairness, ethical considerations, and compliance with regulations like anti-discrimination laws.

# Data Structure and Initial Checks:
The data consists of four tables: Loan Information, Personal Information, Employment, and Other Information, with a total row count of 143,727 records. 

![ER Diagram](https://github.com/manmeetkaurbaxi/loan-default-prediction/blob/main/Images/data_structure_diagram.png)

Prior to performing any predictions, a variety of checks were conducted for missing values, data skewness, multicollinearity, and data imbalance. The Python code utilized to inspect and perform data quality checks and preparation can be found [here](https://github.com/manmeetkaurbaxi/loan-default-prediction/blob/main/Notebooks/1.%20Data%20prep%20and%20ML.ipynb). 

# Assumptions:
Throughout the analysis and model-building process, multiple assumptions were made to manage challenges with the data. These are noted below:
1. For **missing values** in the columns: Social profiles, Is verified, Married, Industry, Work Experience, Amount, Employment Type, Tier of employment, each was handled appropriately.
   - A new category for NA values was created for the column's 'Social Profile', 'Is verified', 'Employment Type' and 'Married'.
   - Missing values in the columns 'Industry' and 'Work Experience' were dropped.
   - Assume that null values in the 'Amount' column are replaced with "-1000" to differentiate them, and a new column "amount_missing" is created to flag missing values (1 if null, 0 otherwise).
   - Null values in the 'Tier of Employment' column are replaced with the string "Z" to categorize them separately.
2. Categorical columns with many categories, such as 'Industry,' 'Role,' and 'User ID,' are dropped to prevent excessive model dimensionality.
3. Considering privacy concerns, the 'Pincode' column is dropped from the dataset. It can be though converted into latitude and longitude values to enhance the analysis potentially.
4. Multiple variables, including 'Amount,' 'Payment,' 'Received Principal,' and 'Interest Received,' exhibit right-skewed distributions, while variables like 'Employment Type' and 'Work Experience' show imbalances. Z-score is applied to handle data skewness.
5. Oversampling (SMOTE) is applied to the minority class (defaulters) by duplicating or generating synthetic examples to balance the dataset.
