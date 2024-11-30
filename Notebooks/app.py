import streamlit as st
import pandas as pd
from io import StringIO
import warnings
from Source.setup import read_excel_data
from Source.eda import (correlation_heatmap, 
                       plot_pairwise_scatter)
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import shap

# Suppress warnings
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, message='.*?pyplot.*?')

def load_model():
    with open('../Output/xgboost_model.pkl', 'rb') as file:
        return pickle.load(file)
    
def main():
    st.set_page_config(page_title = 'Loan Default Prediction',
                   page_icon = '💸',
                   layout = 'wide')
    st.title("Loan Default Prediction")
    st.markdown('##### Made with 🍵 by Manmeet Kaur Baxi | [Portfolio](https://manmeetkaurbaxi.com)')
    
    # Data Loading Section with Progress Bar
    if 'data' not in st.session_state:
        with st.spinner('Loading raw data...'):
            path = '../Input/raw/Credit_Risk_Dataset.xlsx'
            sheet_names = ['loan_information', 'Employment', 'Personal_information', 'Other_information']
            
            dfs = read_excel_data(path, sheet_names)
            
            # Merge all dataframes
            loan_information = dfs[0]
            employment = dfs[1]
            personal_information = dfs[2]
            other_information = dfs[3]
            
            # Merging dataframes
            merged_df = pd.merge(loan_information, employment, left_on='User_id', right_on='User id')
            merged_df = pd.merge(merged_df, personal_information, left_on='User_id', right_on='User id')
            merged_df = pd.merge(merged_df, other_information, left_on='User_id', right_on='User_id')
            
            st.session_state['data'] = merged_df
            st.success("Raw data loaded successfully!")        

    # Load cleaned data and model if not in session state
    if 'cleaned_data' not in st.session_state or 'model' not in st.session_state:
        with st.spinner('Loading cleaned data and model...'):
            st.session_state['cleaned_data'] = pd.read_excel('../Input/cleaned/CleanedData.xlsx')
            st.session_state['model'] = load_model()
            st.success("Cleaned data and model loaded successfully!")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Choose Page",
        ["EDA", "Model Performance", "Prediction Form"]
    )
    
    if page == "EDA":
        # EDA Section
        if 'data' in st.session_state:
            eda_option = st.sidebar.selectbox(
                "Choose EDA Analysis",
                ["Data Overview", "Correlation Analysis", "Pairwise Scatter Plots", "Skewness Analysis", "Class Imbalance"]
            )
            
            if eda_option == "Data Overview":
                st.subheader("Dataset Overview")
                st.write("Shape of the dataset:", st.session_state['data'].shape)
                st.write("Sample of the dataset:")
                st.dataframe(st.session_state['data'].head())
                st.write("Data Info:")
                buffer = StringIO()
                st.session_state['data'].info(buf=buffer)
                st.text(buffer.getvalue())
                
            elif eda_option == "Correlation Analysis":
                st.subheader("Correlation Heatmap")
                fig = correlation_heatmap(st.session_state['data'].select_dtypes(include=['float64', 'int64']))
                st.pyplot(fig)
                
            elif eda_option == "Pairwise Scatter Plots":
                st.subheader("Pairwise Scatter Plots")
                numeric_cols = st.multiselect(
                    "Select features for scatter plot (max 3 recommended)",
                    st.session_state['data'].select_dtypes(include=['float64', 'int64']).columns
                )
                if numeric_cols:
                    fig = plot_pairwise_scatter(st.session_state['data'][numeric_cols])
                    st.pyplot(fig)
                
            elif eda_option == "Skewness Analysis":
                st.subheader("Skewness Analysis")
                numeric_data = st.session_state['data'].select_dtypes(include=['float64', 'int64'])
                skewness_features = ['Amount','Interest Rate','Tenure(years)','Dependents','Total Payement ','Received Principal','Interest Received']
                skewness = numeric_data[skewness_features].skew()
                
                st.write("Skewness Values:")
                st.write(skewness)
                
                feature = st.selectbox("Select feature to visualize distribution", skewness_features)
                fig, ax = plt.subplots()
                sns.histplot(data=numeric_data, x=feature, kde=True, ax=ax)
                plt.title(f'Distribution of {feature}')
                st.pyplot(fig)
                
            elif eda_option == "Class Imbalance":
                st.subheader("Class Imbalance Analysis")
                if 'Defaulter' in st.session_state['data'].columns:
                    fig, ax = plt.subplots()
                    sns.countplot(data=st.session_state['data'], x='Defaulter', ax=ax)
                    plt.title('Class Distribution')
                    st.pyplot(fig)
                    
                    class_proportions = st.session_state['data']['Defaulter'].value_counts(normalize=True)
                    st.write("Class Proportions:")
                    st.write(class_proportions)
                else:
                    st.write("Target variable 'Defaulter' not found in the dataset")
    
    elif page == "Model Performance":
        st.subheader("Model Performance on Cleaned Data")
        X = st.session_state['cleaned_data'].drop('Defaulter', axis=1)
        y = st.session_state['cleaned_data']['Defaulter']
        y_pred = st.session_state['model'].predict(X)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Classification Report:")
            report = classification_report(y, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
        with col2:
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots(figsize=(2, 1))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={'size': 3})
            plt.ylabel('True Label', fontsize=4)
            plt.xlabel('Predicted Label', fontsize=4)
            plt.tick_params(labelsize=4)
            plt.tight_layout()
            st.pyplot(fig)
        
        # SHAP Analysis
        st.subheader("SHAP Analysis")
        
        with st.spinner('Calculating SHAP values...'):
            explainer = shap.TreeExplainer(st.session_state['model'])
            shap_values = explainer.shap_values(X)
            
            # Detailed Summary Plot
            st.write("SHAP Summary Plot")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(fig)
            plt.clf()
            
            # Sample SHAP Force Plot
            st.write("SHAP Force Plot for a Sample Prediction")
            sample_idx = 0
            st.pyplot(shap.force_plot(explainer.expected_value, 
                                    shap_values[sample_idx,:], 
                                    X.iloc[sample_idx,:], 
                                    matplotlib=True,
                                    show=False))
        
    elif page == "Prediction Form":
        st.subheader("Loan Default Prediction Form")
        
        # Example inputs
        st.write("Example inputs:")
        st.write("Likely Defaulter: Amount=80000, Interest Rate=15.5, Tenure=5, Work Experience=1, Total Income=35000, Dependents=3")
        # st.write("Likely Non-Defaulter: Amount=30000, Interest Rate=10.5, Tenure=2, Work Experience=5, Total Income=65000, Dependents=2")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.number_input("Loan Amount", min_value=0.0)
                interest_rate = st.number_input("Interest Rate", min_value=0.0, max_value=100.0)
                tenure = st.number_input("Tenure (years)", min_value=0)                
                
            with col2:
                total_income = st.number_input("Total Income (PA)", min_value=0.0)
                dependents = st.number_input("Number of Dependents", min_value=0)
                work_experience = st.number_input("Work Experience (years)", min_value=0)
                
            submitted = st.form_submit_button("Predict")
            
            if submitted:
                # Create a DataFrame with all required columns initialized to 0
                input_data = pd.DataFrame(columns=['Amount', 'Interest Rate', 'Tenure(years)', 'Tier of Employment', 
                                                 'Work Experience', 'Total Income(PA)', 'Dependents', 'Delinq_2yrs', 
                                                 'Total Payement ', 'Received Principal', 'Interest Received', 
                                                 'Number of loans', 'amount_missing', 'Gender_Female', 'Gender_Male', 
                                                 'Gender_Other', 'Married_No', 'Married_Yes', 'Married_missing', 
                                                 'Home_mortgage', 'Home_none', 'Home_other', 'Home_own', 'Home_rent', 
                                                 'Social Profile_No', 'Social Profile_Yes', 'Social Profile_missing', 
                                                 'Loan Category_Business', 'Loan Category_Car ', 'Loan Category_Consolidation', 
                                                 'Loan Category_Credit Card', 'Loan Category_Home', 'Loan Category_Medical ', 
                                                 'Loan Category_Other ', 'Employmet type_Salaried', 
                                                 'Employmet type_Self - Employeed', 'Employmet type_missing', 
                                                 'Is_verified_Not Verified', 'Is_verified_Source Verified', 
                                                 'Is_verified_Verified', 'Is_verified_missing'])
                
                # Fill with zeros first
                input_data.loc[0] = 0
                
                # Update with actual input values
                input_data.loc[0, 'Amount'] = float(amount)
                input_data.loc[0, 'Interest Rate'] = float(interest_rate)
                input_data.loc[0, 'Tenure(years)'] = float(tenure)
                input_data.loc[0, 'Work Experience'] = float(work_experience)
                input_data.loc[0, 'Total Income(PA)'] = float(total_income)
                input_data.loc[0, 'Dependents'] = float(dependents)
                
                prediction = st.session_state['model'].predict(input_data)
                probability = st.session_state['model'].predict_proba(input_data)
                
                st.write("Prediction:", "Defaulter" if prediction[0] == 1 else "Non-Defaulter")
                st.write(f"Probability of Default: {probability[0][1]:.2%}")

if __name__ == "__main__":
    main()