# Project Background
The Loan Default Prediction project addresses the critical challenge of assessing loan default risks for financial institutions. This initiative analyzes historical borrower data, including financial history, employment details, and past loan performance, to identify patterns and insights. The primary objective is to leverage machine learning to enhance decision-making, balance risk mitigation, and maximize lending opportunities while ensuring transparency and compliance.

Insights and recommendations are provided on the following key areas:
- **Cost Analysis**: Evaluating the financial impact of false positives (denying loans to creditworthy individuals) and false negatives (approving loans for likely defaulters).
- **Bias and Fairness**: Ensuring the model does not exhibit bias against protected attributes such as age or gender, maintaining compliance with regulatory requirements.
- **Global and Local Model Interpretability**: Utilizing SHAP values to understand feature importance at global and individual levels, aiding transparency for data scientists, stakeholders, and regulators.
- **Explainability Methods**: Apply techniques such as anchors and counterfactuals to provide clear, rule-based explanations for model predictions and address the concerns of stakeholders and regulators.

The Python notebook utilized to inspect and perform data quality checks and preparation can be found [here](https://github.com/manmeetkaurbaxi/loan-default-prediction/blob/main/Notebooks/1.%20Data%20prep%20and%20ML.ipynb). 

The Python notebook for the aforementioned insights and recommendations can be found [here](https://github.com/manmeetkaurbaxi/loan-default-prediction/blob/main/Notebooks/2.%20Explainable%20AI.ipynb).

# Data Structure and Initial Checks
The data consists of four tables: Loan Information, Personal Information, Employment, and Other Information, with a total row count of 143,727 records. 

![ER Diagram](https://github.com/manmeetkaurbaxi/loan-default-prediction/blob/main/Images/data_structure_diagram.png)

Prior to performing any predictions, a variety of checks were conducted for missing values, data skewness, multicollinearity, and data imbalance. The Python code utilized to inspect and perform data quality checks and preparation can be found [here](https://github.com/manmeetkaurbaxi/loan-default-prediction/blob/main/Notebooks/1.%20Data%20prep%20and%20ML.ipynb). 

# Insights Deep Dive
The project successfully identified strategies to minimize financial risks while ensuring fairness and transparency in loan decision-making.
##  Cost Analysis
- The cost of false negatives (approving loans for potential defaulters) is $331.5 million, significantly outweighing the $44.84 million cost of false positives (denying loans to non-defaulters).
- Priority should be on reducing false negatives to minimize financial risk while maintaining precision and recall for non-defaulters.
- _Assumptions for cost analysis:_
  - **Cost of False Positives:** $10,000 (Let's assume that the average amount of loan given to the customer is more than $10,000 and a lending institution will lose $10,000 on average if someone doesn’t default. This is the cost associated with incorrectly classifying a loan as default).
  - **Cost of False Negatives:** $50,000 (Let's assume that the average amount of loan given to the customer is more than $50,000, and a lending institution will lose $50,000 on average if someone defaults. This is the cost associated with incorrectly classifying a loan as non-default).
    
![confusion matrix](https://github.com/manmeetkaurbaxi/loan-default-prediction/blob/main/Images/confusion%20matrix.jpg)

## Bias and Fairness
DeepChecks analysis revealed no evidence of bias based on attributes like age or gender within the false positive segment. Weak segments contributing to misclassifications were linked to features such as "Received Principal vs Amount", "Received Principal vs Interest Rate".

![principal vs amount](https://github.com/manmeetkaurbaxi/loan-default-prediction/blob/main/Images/principal_vs_amount.jpg)
![principal_vs_interest](https://github.com/manmeetkaurbaxi/loan-default-prediction/blob/main/Images/principal_vs_interest.jpg)

## Global and Local Model Interpretability
- SHAP analysis highlighted key factors influencing loan default predictions, such as "Received Principal" and "Credit Score."
- Local explanations enabled data scientists to illustrate feature impacts for individual predictions, enhancing trust and collaboration among stakeholders and regulators.

## Explainability Methods
- Anchors provided rule-based insights into decisions, such as identifying critical thresholds for approval or denial.
- Counterfactuals offered scenarios demonstrating how altering specific features could change outcomes, ensuring fairness and transparency in the model’s decision-making process.

# Recommendations
1. Enhance Model Performance: Focus on reducing false negatives through fine-tuning, feature optimization, and incorporating additional data sources to improve recall and precision for non-defaulters.
2. Address Bias and Fairness: Regularly audit the model using tools like DeepChecks to ensure compliance with anti-discrimination laws and fair lending practices.
3. Improve Interpretability: Leverage SHAP values to continuously monitor and explain feature importance, ensuring clarity for both technical and non-technical stakeholders.
4. Adopt Explainability Techniques:
   - Use Anchors to provide straightforward, rule-based explanations for decisions, helping stakeholders understand the rationale behind approvals and denials.
   - Apply Counterfactuals to demonstrate the model’s fairness and robustness, addressing regulatory concerns and building confidence in the decision-making process.
     
By implementing these recommendations, financial institutions can ensure a robust, transparent, and fair loan decision-making framework that aligns with business goals and regulatory standards.

# Assumptions
Throughout the analysis and model-building process, multiple assumptions were made to manage challenges with the data. These are noted below:
1. For **missing values** in the columns: Social profiles, Is verified, Married, Industry, Work Experience, Amount, Employment Type, Tier of employment, each was handled appropriately.
   - A new category for NA values was created for the column's 'Social Profile', 'Is verified', 'Employment Type' and 'Married'.
   - Missing values in the columns 'Industry' and 'Work Experience' were dropped.
   - Null values in the 'Amount' column are replaced with "-1000" to differentiate them, and a new column, "amount_missing", is created to flag missing values (1 if null, 0 otherwise).
   - Null values in the 'Tier of Employment' column are replaced with the string "Z" to categorize them separately.
2. Categorical columns with many categories, such as 'Industry,' 'Role,' and 'User ID,' are dropped to prevent excessive model dimensionality.
3. Considering privacy concerns, the 'Pincode' column is dropped from the dataset. It can be though converted into latitude and longitude values to enhance the analysis geospatially.
4. Multiple variables, including 'Amount,' 'Payment,' 'Received Principal,' and 'Interest Received,' exhibit right-skewed distributions, while variables like 'Employment Type' and 'Work Experience' show imbalances. Z-score is applied to handle data skewness.
5. Oversampling (SMOTE) is applied to the minority class (defaulters) by duplicating or generating synthetic examples to balance the dataset.

_Please give a 🌟 if you found this repository helpful in any manner._
