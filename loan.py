import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import plotly.express as px
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Setting up page configuration

st.set_page_config(page_title= "Financial Risk Detection",
                   layout= "wide",
                   initial_sidebar_state= "expanded"                   
                  )

# Creating Background

def setting_bg():
    st.markdown(f""" <style>.stApp {{
                    background:url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQusP3jn05fepU72kFxDCVTboLA6_tlJdZIxw&usqp=CAU");
                    background-size: cover}}
                </style>""", unsafe_allow_html=True)
setting_bg()

# Creating option menu in the side bar

with st.sidebar:

    selected = option_menu("Menu", ["Home","Prediction","EDA Analysis", "Exploration"], 
                           icons=["house","list-task","bar-chart-line","star"],
                           menu_icon= "menu-button-wide",
                           default_index=0,
                           styles={"nav-link": {"font-size": "20px", "text-align": "left", "margin": "-2px", "--hover-color": "blue"},
                                   "nav-link-selected": {"background-color": "blue"}}
                          )
    
# Home Menu

if selected == 'Home':

    st.markdown(f'<h1 style="text-align: center; color: green;">FINANCIAL RISK DETECTION</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("## :violet[*Domain :*] Financial Loan Risk")
        col1.markdown("# ")
        col1.markdown("## :violet[*Technologies used :*]")
        col1.markdown("##  Python, Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, Streamlit. ")
        col1.markdown("# ")
        col1.markdown("## :violet[*Overview :*]")
        col1.markdown("##   Build Classification Model to Predict financial risk ")
        col1.markdown("# ")

    with col2:
        col2.markdown("# ")
        col2.markdown("# ")
        col2.image("/Users/arul/Downloads/risk1.jpg")
        col2.markdown("# ")
        col2.image("/Users/arul/Downloads/risk2.jpg")

NAME_CONTRACT_TYPE_CURR = {0 : 'Cash loans', 1 : 'Revolving loans'}

CODE_GENDER = {0 : 'F', 1 : 'M', 2 : 'XNA'}

FLAG_OWN_CAR = {0 : 'N', 1 : 'Y'}

FLAG_OWN_REALTY = {0 : 'N', 1 : 'Y'}

NAME_TYPE_SUITE = {0 : 'Children', 1 : 'Family', 2 : 'Group of people', 3 : 'Other_A', 4 : 'Other_B', 5 : 'Spouse, partner', 6 : 'Unaccompanied'}

NAME_INCOME_TYPE = {0 : 'Commercial associate', 1 : 'Maternity leave', 2 : 'Pensioner', 3 : 'State servant', 4 : 'Student', 5 : 'Unemployed', 6 : 'Working'}

NAME_EDUCATION_TYPE = {0 : 'Academic degree', 1 : 'Higher education', 2 : 'Incomplete higher', 3 : 'Lower secondary', 4 : 'Secondary / secondary special'}

NAME_FAMILY_STATUS = {0 : 'Civil marriage', 1 : 'Married', 2 : 'Separated', 3 : 'Single / not married', 4 : 'Widow'}

NAME_HOUSING_TYPE = {0 : 'Co-op apartment', 1 : 'House / apartment', 2 : 'Municipal apartment', 3 : 'Office apartment', 4 : 'Rented apartment', 5 : 'With parents'}

OCCUPATION_TYPE = {0 : 'Accountants', 1 : 'Cleaning staff', 2 : 'Cooking staff', 3 : 'Core staff', 4 : 'Drivers', 5 : 'HR staff', 6 : 'High skill tech staff', 7 : 'IT staff', 8 : 'Laborers', 9 : 'Low-skill Laborers', 10 : 'Managers', 11 : 'Medicine staff', 12 : 'Private service staff', 13 : 'Realty agents', 14 : 'Sales staff', 15 : 'Secretaries', 16 : 'Security staff', 17 : 'Unknown', 18 : 'Waiters/barmen staff'}

ORGANIZATION_TYPE = {0 : 'Advertising', 1 : 'Agriculture', 2 : 'Bank', 3 : 'Business Entity Type 1', 4 : 'Business Entity Type 2', 5 : 'Business Entity Type 3', 6 : 'Cleaning', 7 : 'Construction', 8 : 'Culture', 9 : 'Electricity', 10 : 'Emergency', 11 : 'Government', 12 : 'Hotel', 13 : 'Housing', 14 : 'Industry: type 1', 15 : 'Industry: type 10', 16 : 'Industry: type 11', 17 : 'Industry: type 12', 18 : 'Industry: type 13', 19 : 'Industry: type 2', 20 : 'Industry: type 3', 21 : 'Industry: type 4', 22 : 'Industry: type 5', 23 : 'Industry: type 6', 24 : 'Industry: type 7', 25 : 'Industry: type 8', 26 : 'Industry: type 9', 27 : 'Insurance', 28 : 'Kindergarten', 29 : 'Legal Services', 30 : 'Medicine', 31 : 'Military', 32 : 'Mobile', 33 : 'Other', 34 : 'Police', 35 : 'Postal', 36 : 'Realtor', 37 : 'Religion', 38 : 'Restaurant', 39 : 'School', 40 : 'Security', 41 : 'Security Ministries', 42 : 'Self-employed', 43 : 'Services', 44 : 'Telecom', 45 : 'Trade: type 1', 46 : 'Trade: type 2', 47 : 'Trade: type 3', 48 : 'Trade: type 4', 49 : 'Trade: type 5', 50 : 'Trade: type 6', 51 : 'Trade: type 7', 52 : 'Transport: type 1', 53 : 'Transport: type 2', 54 : 'Transport: type 3', 55 : 'Transport: type 4', 56 : 'University', 57 : 'XNA'}

AMT_INCOME_RANGE = {0 : '0-100K', 1 : '100K-200K', 2 : '1M Above', 3 : '200k-300k', 4 : '300k-400k', 5 : '400k-500k', 6 : '500k-600k', 7 : '600k-700k', 8 : '700k-800k', 9 : '800k-900k', 10 : '900k-1M'}

AMT_CREDIT_RANGE = {0 : '0-100K', 1 : '100K-200K', 2 : '1M Above', 3 : '200k-300k', 4 : '300k-400k', 5 : '400k-500k', 6 : '500k-600k', 7 : '600k-700k', 8 : '700k-800k', 9 : '800k-900k', 10 : '900k-1M'}

AGE_GROUP = {0 : '0-20', 1 : '20-30', 2 : '30-40', 3 : '40-50', 4 : '50 above'}

NAME_CONTRACT_TYPE_y = {0 : 'Cash loans', 1 : 'Consumer loans', 2 : 'Revolving loans'}

NAME_CASH_LOAN_PURPOSE = {0 : 'Building a house or an annex', 1 : 'Business development', 2 : 'Buying a garage', 3 : 'Buying a holiday home / land', 4 : 'Buying a home', 5 : 'Buying a new car', 6 : 'Buying a used car', 7 : 'Car repairs', 8 : 'Education', 9 : 'Everyday expenses', 10 : 'Furniture', 11 : 'Gasification / water supply', 12 : 'Hobby', 13 : 'Journey', 14 : 'Medicine', 15 : 'Money for a third person', 16 : 'Other', 17 : 'Payments on other loans', 18 : 'Purchase of electronic equipment', 19 : 'Refusal to name the goal', 20 : 'Repairs', 21 : 'Urgent needs', 22 : 'Wedding / gift / holiday', 23 : 'XAP', 24 : 'XNA'}

NAME_CONTRACT_STATUS = {0 : 'Approved', 1 : 'Canceled', 2 : 'Refused', 3 : 'Unused offer'}

NAME_PAYMENT_TYPE = {0 : 'Cash through the bank', 1 : 'Cashless from the account of the employer', 2 : 'Non-cash from your account', 3 : 'XNA'}

CODE_REJECT_REASON = {0 : 'CLIENT', 1 : 'HC', 2 : 'LIMIT', 3 : 'SCO', 4 : 'SCOFR', 5 : 'SYSTEM', 6 : 'VERIF', 7 : 'XAP', 8 : 'XNA'}

NAME_CLIENT_TYPE = {0 : 'New', 1 : 'Refreshed', 2 : 'Repeater', 3 : 'XNA'}

NAME_GOODS_CATEGORY = {0 : 'Additional Service', 1 : 'Animals', 2 : 'Audio/Video', 3 : 'Auto Accessories', 4 : 'Clothing and Accessories', 5 : 'Computers', 6 : 'Construction Materials', 7 : 'Consumer Electronics', 8 : 'Direct Sales', 9 : 'Education', 10 : 'Fitness', 11 : 'Furniture', 12 : 'Gardening', 13 : 'Homewares', 14 : 'Insurance', 15 : 'Jewelry', 16 : 'Medical Supplies', 17 : 'Medicine', 18 : 'Mobile', 19 : 'Office Appliances', 20 : 'Other', 21 : 'Photo / Cinema Equipment', 22 : 'Sport and Leisure', 23 : 'Tourism', 24 : 'Vehicles', 25 : 'Weapon', 26 : 'XNA'}

NAME_PORTFOLIO = {0 : 'Cards', 1 : 'Cars', 2 : 'Cash', 3 : 'POS', 4 : 'XNA'}

NAME_PRODUCT_TYPE = {0 : 'XNA', 1 : 'walk-in', 2 : 'x-sell'}

CHANNEL_TYPE = {0 : 'AP+ (Cash loan)', 1 : 'Car dealer', 2 : 'Channel of corporate sales', 3 : 'Contact center', 4 : 'Country-wide', 5 : 'Credit and cash offices', 6 : 'Regional / Local', 7 : 'Stone'}

NAME_SELLER_INDUSTRY = {0 : 'Auto technology', 1 : 'Clothing', 2 : 'Connectivity', 3 : 'Construction', 4 : 'Consumer electronics', 5 : 'Furniture', 6 : 'Industry', 7 : 'Jewelry', 8 : 'MLM partners', 9 : 'Tourism', 10 : 'XNA'}

NAME_YIELD_GROUP = {0 : 'XNA', 1 : 'high', 2 : 'low_action', 3 : 'low_normal', 4 : 'middle'}

PRODUCT_COMBINATION = {0 : 'Card Street', 1 : 'Card X-Sell', 2 : 'Cash', 3 : 'Cash Street: high', 4 : 'Cash Street: low', 5 : 'Cash Street: middle', 6 : 'Cash X-Sell: high', 7 : 'Cash X-Sell: low', 8 : 'Cash X-Sell: middle', 9 : 'POS household with interest', 10 : 'POS household without interest', 11 : 'POS industry with interest', 12 : 'POS industry without interest', 13 : 'POS mobile with interest', 14 : 'POS mobile without interest', 15 : 'POS other with interest', 16 : 'POS others without interest'}

if selected == 'Prediction':
    with st.form("form1"):

            col1,col2,col3 = st.columns([0.45, 0.1, 0.45])

            with col1:
                  SK_ID_CURR = st.number_input('ID Current Loan', min_value=100002, max_value=456255 ,value=100002 )
                  CNT_CHILDREN = st.number_input('Number of Children', min_value=0, max_value=19 ,value=0)
                  AMT_INCOME_TOTAL = st.number_input('Total Income', min_value=0.2565, max_value=10.8 ,value=0.2565)
                  AMT_CREDIT_CURR = st.number_input('Current Loan Amount', min_value=0.45, max_value=39.56274 ,value=0.45)

                  AMT_ANNUITY_CURR = st.number_input('Current Annuity', min_value=1615.5, max_value=225000.0 ,value=1615.5)
                  AMT_GOODS_PRICE_CURR = st.number_input('Current Goods Price', min_value=40500.0, max_value=3825000.0 ,value=40500.0)
                  DAYS_BIRTH = st.number_input('Age In Days', min_value=7489, max_value=25201 ,value=7489)
                  DAYS_EMPLOYED = st.number_input('Employed Days', min_value=0, max_value=365243 ,value=0)

                  DAYS_REGISTRATION = st.number_input('Registration Days', min_value=0.0, max_value=24672.0 ,value=0.0)
                  DAYS_ID_PUBLISH = st.number_input('ID Publish Days', min_value=0, max_value=7197 ,value=0)
                  CNT_FAM_MEMBERS = st.number_input('Family Members', min_value=1.0, max_value=20.0 ,value=1.0)
                  FLAG_DOCUMENT_3 = st.number_input('Document 3', min_value=0, max_value=1 ,value=0)

                  SK_ID_PREV = st.number_input('ID Previous Loan', min_value=1000001, max_value=2845381 ,value=1000001)
                  AMT_ANNUITY_PREV = st.number_input('Previous Annuity', min_value=0.0, max_value=418058.145 ,value=0.0)
                  AMT_APPLICATION = st.number_input('Application Amount', min_value=0.0, max_value=5850000.0 ,value=0.0)
                  AMT_CREDIT_PREV = st.number_input('Previous Credit', min_value=0.0, max_value=4509688.5 ,value=0.0)

                  AMT_GOODS_PRICE_PREV = st.number_input('Previous Goods Price', min_value=0.0, max_value=5850000.0 ,value=0.0)
                  DAYS_DECISION = st.number_input('Decision Days', min_value=1, max_value=2922 ,value=1)
                  CNT_PAYMENT = st.number_input('Payment', min_value=0.0, max_value=84.0 ,value=0.0)

                  NAME_CONTRACT_TYPE_CURR = st.selectbox("Contract Type", options=NAME_CONTRACT_TYPE_CURR.keys(), format_func=lambda x: NAME_CONTRACT_TYPE_CURR[x])
                  CODE_GENDER = st.selectbox("Gender", options=CODE_GENDER.keys(), format_func=lambda x: CODE_GENDER[x])
                  FLAG_OWN_CAR = st.selectbox("Car", options=FLAG_OWN_CAR.keys(), format_func=lambda x: FLAG_OWN_CAR[x])
                  FLAG_OWN_REALTY = st.selectbox("Realty", options=FLAG_OWN_REALTY.keys(), format_func=lambda x: FLAG_OWN_REALTY[x])

            with col3:
                #  device_deviceCategory = st.selectbox('DEVICE CATEGORY', options=device_categories.keys(), format_func=lambda x: device_categories[x])
                

                NAME_TYPE_SUITE = st.selectbox("Type Suite", options=NAME_TYPE_SUITE.keys(), format_func=lambda x: NAME_TYPE_SUITE[x])
                NAME_INCOME_TYPE = st.selectbox("Income Type", options=NAME_INCOME_TYPE.keys(), format_func=lambda x: NAME_INCOME_TYPE[x])
                NAME_EDUCATION_TYPE = st.selectbox("Education", options=NAME_EDUCATION_TYPE.keys(), format_func=lambda x: NAME_EDUCATION_TYPE[x])
                NAME_FAMILY_STATUS = st.selectbox("Family", options=NAME_FAMILY_STATUS.keys(), format_func=lambda x: NAME_FAMILY_STATUS[x])

                NAME_HOUSING_TYPE = st.selectbox("Housing", options=NAME_HOUSING_TYPE.keys(), format_func=lambda x: NAME_HOUSING_TYPE[x])
                OCCUPATION_TYPE = st.selectbox("Occupation", options=OCCUPATION_TYPE.keys(), format_func=lambda x: OCCUPATION_TYPE[x])
                ORGANIZATION_TYPE = st.selectbox("Organization", options=ORGANIZATION_TYPE.keys(), format_func=lambda x: ORGANIZATION_TYPE[x])
                AMT_INCOME_RANGE = st.selectbox("Income Range", options=AMT_INCOME_RANGE.keys(), format_func=lambda x: AMT_INCOME_RANGE[x])

                AMT_CREDIT_RANGE = st.selectbox("Credit Range", options=AMT_CREDIT_RANGE.keys(), format_func=lambda x: AMT_CREDIT_RANGE[x])
                AGE_GROUP = st.selectbox("Age Group", options=AGE_GROUP.keys(), format_func=lambda x: AGE_GROUP[x])
                NAME_CONTRACT_TYPE_y = st.selectbox("Contract Type", options=NAME_CONTRACT_TYPE_y.keys(), format_func=lambda x: NAME_CONTRACT_TYPE_y[x])
                NAME_CASH_LOAN_PURPOSE = st.selectbox("Purpose", options=NAME_CASH_LOAN_PURPOSE.keys(), format_func=lambda x: NAME_CASH_LOAN_PURPOSE[x])

                NAME_CONTRACT_STATUS = st.selectbox("Status", options=NAME_CONTRACT_STATUS.keys(), format_func=lambda x: NAME_CONTRACT_STATUS[x])
                NAME_PAYMENT_TYPE = st.selectbox("Payment Type", options=NAME_PAYMENT_TYPE.keys(), format_func=lambda x: NAME_PAYMENT_TYPE[x])
                CODE_REJECT_REASON = st.selectbox("Reject Reason", options=CODE_REJECT_REASON.keys(), format_func=lambda x: CODE_REJECT_REASON[x])
                NAME_CLIENT_TYPE = st.selectbox("Client Type", options=NAME_CLIENT_TYPE.keys(), format_func=lambda x: NAME_CLIENT_TYPE[x])

                NAME_GOODS_CATEGORY = st.selectbox("Goods Category", options=NAME_GOODS_CATEGORY.keys(), format_func=lambda x: NAME_GOODS_CATEGORY[x])
                NAME_PORTFOLIO = st.selectbox("Portfolio", options=NAME_PORTFOLIO.keys(), format_func=lambda x: NAME_PORTFOLIO[x])
                NAME_PRODUCT_TYPE = st.selectbox("Product Type", options=NAME_PRODUCT_TYPE.keys(), format_func=lambda x: NAME_PRODUCT_TYPE[x])
                CHANNEL_TYPE = st.selectbox("Channel Type", options=CHANNEL_TYPE.keys(), format_func=lambda x: CHANNEL_TYPE[x])

                NAME_SELLER_INDUSTRY = st.selectbox("Industry", options=NAME_SELLER_INDUSTRY.keys(), format_func=lambda x: NAME_SELLER_INDUSTRY[x])
                NAME_YIELD_GROUP = st.selectbox("Yield Group", options=NAME_YIELD_GROUP.keys(), format_func=lambda x: NAME_YIELD_GROUP[x])
                PRODUCT_COMBINATION = st.selectbox("Product Combination", options=PRODUCT_COMBINATION.keys(), format_func=lambda x: PRODUCT_COMBINATION[x])

            with col1: 
                col1.markdown("# ")
                col1.markdown("# ")
                submit_button = st.form_submit_button("Submit")

                if submit_button is not None:
                    with open('risk.pkl', 'rb') as f:
                        pick_model6 = pickle.load(f)


                        new_sample = np.array(
                                            [[SK_ID_CURR, NAME_CONTRACT_TYPE_CURR, CODE_GENDER, FLAG_OWN_CAR,
                                                FLAG_OWN_REALTY, CNT_CHILDREN, AMT_INCOME_TOTAL,
                                                AMT_CREDIT_CURR, AMT_ANNUITY_CURR, AMT_GOODS_PRICE_CURR,
                                                NAME_TYPE_SUITE, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE,
                                                NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, DAYS_BIRTH,
                                                DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH,
                                                OCCUPATION_TYPE, CNT_FAM_MEMBERS, ORGANIZATION_TYPE,
                                                FLAG_DOCUMENT_3, AMT_INCOME_RANGE, AMT_CREDIT_RANGE, AGE_GROUP,
                                                SK_ID_PREV, NAME_CONTRACT_TYPE_y, AMT_ANNUITY_PREV,
                                                AMT_APPLICATION, AMT_CREDIT_PREV, AMT_GOODS_PRICE_PREV,
                                                NAME_CASH_LOAN_PURPOSE, NAME_CONTRACT_STATUS, DAYS_DECISION,
                                                NAME_PAYMENT_TYPE, CODE_REJECT_REASON, NAME_CLIENT_TYPE,
                                                NAME_GOODS_CATEGORY, NAME_PORTFOLIO, NAME_PRODUCT_TYPE,
                                                CHANNEL_TYPE, NAME_SELLER_INDUSTRY, CNT_PAYMENT,
                                                NAME_YIELD_GROUP, PRODUCT_COMBINATION]])                       
                        new_pred = pick_model6.predict(new_sample)[0]
                        
                        if new_pred == 1.0:                       
                            st.markdown(
                                "<h1 style='font-size: 40px;'><span style='color: green;'>Repayer </span></h1>",
                                unsafe_allow_html=True)

                        elif new_pred == 0.0:
                            #print(x)
                            st.markdown(
                                "<h1 style='font-size: 40px;'><span style='color: red;'>Defaulter</span> </h1>",
                                unsafe_allow_html=True)

if selected == 'EDA Analysis':
    st.title(":green[*EDA Analysis*]")

    applicationDF = pd.read_csv('application_df.csv')
    previousDF = pd.read_csv('previous_df.csv')

    tab1,tab2,tab3,tab4, tab5 = st.tabs(['Univariate Categorical', 'Bivariate Bar', 'Bivariate Rel','Univariate Merged', 'Merged Pointplot'])



    with tab1:
        def univariate_categorical(feature, ylog=False, label_rotation=False, horizontal_layout=True):
            st.title(":green[*Univariate Categorical Analysis*]")
            
            applicationDF = pd.read_csv('application_df.csv')
            
            temp = applicationDF[feature].value_counts()
            df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

            # Calculate the percentage of target=1 per category value
            cat_perc = applicationDF[[feature, 'TARGET']].groupby([feature], as_index=False).mean()
            cat_perc["TARGET"] = cat_perc["TARGET"] * 100
            cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
            
            if horizontal_layout:
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
            else:
                fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 24))
            
            # 1. Subplot 1: Count plot of categorical column
            s1 = sns.countplot(ax=ax1, 
                            x=feature, 
                            data=applicationDF,
                            hue="TARGET",
                            order=cat_perc[feature],
                            palette=['g', 'r'])
            
            # Define common styling
            ax1.set_title(feature, fontdict={'fontsize': 10, 'fontweight': 3, 'color': 'Blue'})
            ax1.legend(['Repayer', 'Defaulter'])
            
            # If the plot is not readable, use the log scale.
            if ylog:
                ax1.set_yscale('log')
                ax1.set_ylabel("Count (log)", fontdict={'fontsize': 10, 'fontweight': 3, 'color': 'Blue'})
            
            if label_rotation:
                s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
            
            # 2. Subplot 2: Percentage of defaulters within the categorical column
            s2 = sns.barplot(ax=ax2, 
                            x=feature, 
                            y='TARGET', 
                            order=cat_perc[feature], 
                            data=cat_perc,
                            palette='Set2')
            
            if label_rotation:
                s2.set_xticklabels(s2.get_xticklabels(), rotation=90)
            
            plt.ylabel('Percent of Defaulters [%]', fontsize=10)
            plt.tick_params(axis='both', which='major', labelsize=10)
            ax2.set_title(feature + " Defaulter %", fontdict={'fontsize': 15, 'fontweight': 5, 'color': 'Blue'})
            
            # Show the plot in Streamlit
            st.pyplot(fig)

        selection = st.radio("Select the Category", ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_HOUSING_TYPE', 'NAME_FAMILY_STATUS','NAME_EDUCATION_TYPE','NAME_INCOME_TYPE','REGION_RATING_CLIENT','OCCUPATION_TYPE','ORGANIZATION_TYPE','FLAG_DOCUMENT_3','AGE_GROUP','EMPLOYMENT_YEAR','AMT_CREDIT_RANGE','AMT_INCOME_RANGE','CNT_CHILDREN','CNT_FAM_MEMBERS'])

        if selection == 'NAME_CONTRACT_TYPE':
            univariate_categorical('NAME_CONTRACT_TYPE', True)

        if selection == 'CODE_GENDER':
            univariate_categorical('CODE_GENDER')

        if selection == 'FLAG_OWN_CAR':
            univariate_categorical('FLAG_OWN_CAR')

        if selection == 'FLAG_OWN_REALTY':
            univariate_categorical('FLAG_OWN_REALTY')

        if selection == 'NAME_HOUSING_TYPE':
            univariate_categorical("NAME_HOUSING_TYPE",True,True,True)

        if selection == 'NAME_FAMILY_STATUS':
            univariate_categorical("NAME_FAMILY_STATUS",False,True,True)

        if selection == 'NAME_EDUCATION_TYPE':
            univariate_categorical("NAME_EDUCATION_TYPE",True,True,True)

        if selection == 'NAME_INCOME_TYPE':
            univariate_categorical("NAME_INCOME_TYPE",True,True,False)

        if selection == 'REGION_RATING_CLIENT':
            univariate_categorical("REGION_RATING_CLIENT",False,False,True)

        if selection == 'OCCUPATION_TYPE':
            univariate_categorical("OCCUPATION_TYPE",False,True,False)

        if selection == 'ORGANIZATION_TYPE':
            univariate_categorical("ORGANIZATION_TYPE",True,True,False)

        if selection == 'FLAG_DOCUMENT_3':
            univariate_categorical("FLAG_DOCUMENT_3",False,False,True)

        if selection == 'AGE_GROUP':
            univariate_categorical("AGE_GROUP",False,False,True)

        if selection == 'EMPLOYMENT_YEAR':
            univariate_categorical("EMPLOYMENT_YEAR",False,False,True)

        if selection == 'AMT_CREDIT_RANGE':
            univariate_categorical("AMT_CREDIT_RANGE",False,False,False)

        if selection == 'AMT_INCOME_RANGE':
            univariate_categorical("AMT_INCOME_RANGE",False,False,False)

        if selection == 'CNT_CHILDREN':
            univariate_categorical("CNT_CHILDREN",True)

        if selection == 'CNT_FAM_MEMBERS':
            univariate_categorical("CNT_FAM_MEMBERS",True, False, False)

    with tab2:
        # Set the deprecation option to suppress the warning
        st.set_option('deprecation.showPyplotGlobalUse', False)

        def bivariate_bar(x, y, df, hue, figsize):
            st.title(":green[*Bivariate Bar Analysis*]")

            fig, ax = plt.subplots(figsize=figsize)
            sns.barplot(x=x, y=y, data=df, hue=hue, palette=['g', 'r'], ax=ax)

            # Defining aesthetics of Labels and Title of the plot using style dictionaries
            plt.xlabel(x, fontdict={'fontsize': 10, 'fontweight': 3, 'color': 'Blue'})
            plt.ylabel(y, fontdict={'fontsize': 10, 'fontweight': 3, 'color': 'Blue'})
            plt.title(f"{x} vs {y} with Hue: {hue}", fontdict={'fontsize': 15, 'fontweight': 5, 'color': 'Blue'})
            plt.xticks(rotation=90, ha='right')
            plt.legend(labels=['Repayer', 'Defaulter'])

            # Show the plot in Streamlit
            st.pyplot(fig)

        bivariate_bar("NAME_INCOME_TYPE", "AMT_INCOME_TOTAL", applicationDF, "TARGET", (18, 10))  

    with tab3:
        def bivariate_rel(x, y, data, hue, kind, palette, legend, figsize):
            st.title(":green[*Bivariate Relationship Analysis*]")

            plt.figure(figsize=figsize)
            sns.relplot(x=x,
                        y=y,
                        data=data,
                        hue=hue,
                        kind=kind,
                        palette=['g', 'r'],
                        legend=legend)

            plt.legend(['Repayer', 'Defaulter'])
            plt.xticks(rotation=90, ha='right')

            # Show the plot in Streamlit
            st.pyplot()

        bivariate_rel('AMT_GOODS_PRICE','AMT_CREDIT',applicationDF,"TARGET", "line", ['g','r'], False,(15,6))

    with tab4:
        def univariate_merged(col, df, hue, palette, ylog, figsize):
            st.title(":green[*Univariate Merged Analysis*]")

            plt.figure(figsize=figsize)
            ax = sns.countplot(x=col,
                            data=df,
                            hue=hue,
                            palette=palette,
                            order=df[col].value_counts().index)

            if ylog:
                plt.yscale('log')
                plt.ylabel("Count (log)", fontdict={'fontsize': 10, 'fontweight': 3, 'color': 'Blue'})
            else:
                plt.ylabel("Count", fontdict={'fontsize': 10, 'fontweight': 3, 'color': 'Blue'})

            plt.title(f"{col} Analysis", fontdict={'fontsize': 15, 'fontweight': 5, 'color': 'Blue'})
            plt.legend(loc="upper right")
            plt.xticks(rotation=90, ha='right')

            # Show the plot in Streamlit
            st.pyplot()

        loan_process_df = pd.merge(applicationDF, previousDF, how='inner', on='SK_ID_CURR')

        L0 = loan_process_df[loan_process_df['TARGET']==0] # Repayers
        L1 = loan_process_df[loan_process_df['TARGET']==1] # Defaulters

        univariate_merged("NAME_CASH_LOAN_PURPOSE",L0,"NAME_CONTRACT_STATUS",["#548235","#FF0000","#0070C0","#FFFF00"],True,(18,7))

        univariate_merged("NAME_CASH_LOAN_PURPOSE",L1,"NAME_CONTRACT_STATUS",["#548235","#FF0000","#0070C0","#FFFF00"],True,(18,7))

    with tab5:
        def merged_pointplot(x, y):
            st.title(":green[*Merged Pointplot Analysis*]")
            loan_process_df = pd.merge(applicationDF, previousDF, how='inner', on='SK_ID_CURR')

            plt.figure(figsize=(8, 4))
            sns.pointplot(x=x, y=y, hue="TARGET", data=loan_process_df, palette=['g', 'r'])
            plt.legend(['Repayer', 'Defaulter'])

            # Show the plot in Streamlit
            st.pyplot()

        merged_pointplot("NAME_CONTRACT_STATUS",'AMT_INCOME_TOTAL')

        merged_pointplot("NAME_CONTRACT_STATUS",'DEF_60_CNT_SOCIAL_CIRCLE')        

if selected == "Exploration":

    df = pd.read_csv('final_detection.csv')

    # Streamlit UI
    st.title(":green[Client Profile Analysis]")

    tab1,tab2,tab3,tab4 = st.tabs(['Client Profile Analysis','Financial Analysis','Credit History Analysis','Loan Application Decisions Analysis'])

    with tab1:
        # Default rates across different demographic groups
        demographic_default_rates = df.groupby(['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS']).mean()['TARGET']

        # Educational or employment patterns associated with higher default risk
        education_employment_default = df.groupby(['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE']).mean()['TARGET']

        # Differences in client profiles between those with payment difficulties and those without
        payment_difficulty_profiles = df.groupby('TARGET').mean()
        
        # Create a checkbox
        run_code_checkbox = st.checkbox("Check the Categorical values")

        # Check if the checkbox is selected
        if run_code_checkbox:

            col1, col2 = st.columns(2)
            
            with col1:
                # Mapping for CODE_GENDER
                CODE_GENDER_mapping = {0: 'F', 1: 'M', 2: 'XNA'}

                # Create a DataFrame from the mapping
                df_mapping = pd.DataFrame(list(CODE_GENDER_mapping.items()), columns=['Code', 'Gender'])

                # Display the DataFrame as a table in Streamlit
                st.table(df_mapping)

                # Mapping for NAME_EDUCATION_TYPE
                NAME_EDUCATION_TYPE_mapping = {
                    0: 'Academic degree',
                    1: 'Higher education',
                    2: 'Incomplete higher',
                    3: 'Lower secondary',
                    4: 'Secondary / secondary special'
                }

                # Create a DataFrame from the mapping
                df_mapping_education = pd.DataFrame(list(NAME_EDUCATION_TYPE_mapping.items()), columns=['Code', 'Education Type'])

                # Display the DataFrame as a table in Streamlit
                st.table(df_mapping_education)

                # Mapping for NAME_FAMILY_STATUS
                NAME_FAMILY_STATUS_mapping = {
                    0: 'Civil marriage',
                    1: 'Married',
                    2: 'Separated',
                    3: 'Single / not married',
                    4: 'Widow'
                }

                # Create a DataFrame from the mapping
                df_mapping_family_status = pd.DataFrame(list(NAME_FAMILY_STATUS_mapping.items()), columns=['Code', 'Family Status'])

                # Display the DataFrame as a table in Streamlit
                st.table(df_mapping_family_status)

            with col2:
                # Mapping for OCCUPATION_TYPE
                OCCUPATION_TYPE_mapping = {
                    0: 'Accountants',
                    1: 'Cleaning staff',
                    2: 'Cooking staff',
                    3: 'Core staff',
                    4: 'Drivers',
                    5: 'HR staff',
                    6: 'High skill tech staff',
                    7: 'IT staff',
                    8: 'Laborers',
                    9: 'Low-skill Laborers',
                    10: 'Managers',
                    11: 'Medicine staff',
                    12: 'Private service staff',
                    13: 'Realty agents',
                    14: 'Sales staff',
                    15: 'Secretaries',
                    16: 'Security staff',
                    17: 'Unknown',
                    18: 'Waiters/barmen staff'
                }

                # Create a DataFrame from the mapping
                df_mapping_occupation = pd.DataFrame(list(OCCUPATION_TYPE_mapping.items()), columns=['Code', 'Occupation Type'])

                # Display the DataFrame as a table in Streamlit
                st.table(df_mapping_occupation)

        # Calculate Income to Debt ratio
        df['INCOME_TO_DEBT_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT_CURR']

        # Default rates across different demographic groups
        st.header(":violet[Default Rates Across Different Demographic Groups:]")
        st.write(demographic_default_rates)

        
        # Educational or employment patterns associated with higher default risk
        st.header(":violet[Educational or Employment Patterns Associated with Higher Default Risk:]")
        st.write(education_employment_default)

        # Differences in client profiles between those with payment difficulties and those without
        st.header(":violet[Differences in Client Profiles Between Payment Difficulties and No Difficulties:]")
        st.write(payment_difficulty_profiles)

        # Income to Debt ratio
        st.header(":violet[Income to Debt Ratio:]")
        st.write(df[['SK_ID_CURR', 'INCOME_TO_DEBT_RATIO']])

    with tab2:
        # Relationship between income, debt levels, and loan default probability
        df['INCOME_TO_DEBT_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT_CURR']

        st.header(":violet[Relationship between Income, Debt Levels, and Loan Default Probability:]")
        fig = px.scatter(df, x='INCOME_TO_DEBT_RATIO', y='TARGET', hover_data=['SK_ID_CURR'], title="Income to Debt Ratio vs. Loan Default Probability")
        st.plotly_chart(fig)

        # Income distribution among loan applicants
        st.header(":violet[Income Distribution Among Loan Applicants:]")
        fig = px.histogram(df, x='AMT_INCOME_TOTAL', nbins=30, title="Income Distribution")
        st.plotly_chart(fig)

    with tab3:
        # Relationships between credit risk and reasons for rejection of previous loan application
        st.header(":violet[Relationships between Credit Risk and Reasons for Rejection:]")
        reject_reasons = df.groupby(['TARGET', 'CODE_REJECT_REASON']).size().unstack()

        fig, ax = plt.subplots(figsize=(10, 6))
        reject_reasons.div(reject_reasons.sum(axis=1), axis=0).plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Credit Risk vs. Reasons for Rejection')
        ax.set_ylabel('Proportion')
        ax.set_xlabel('Credit Risk')
        st.pyplot(fig)

    with tab4:
        # Specific loan types, amounts, or purposes correlated with higher default risk
        st.header(":violet[Specific Loan Types, Amounts, and Purposes vs. Default Risk:]")
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

        sns.boxplot(x='TARGET', y='AMT_CREDIT_CURR', data=df, ax=axes[0, 0])
        axes[0, 0].set_title('Loan Amount vs. Default Risk')

        sns.countplot(x='NAME_CONTRACT_TYPE_CURR', hue='TARGET', data=df, ax=axes[0, 1])
        axes[0, 1].set_title('Loan Type vs. Default Risk')

        sns.countplot(x='NAME_CASH_LOAN_PURPOSE', hue='TARGET', data=df, ax=axes[1, 0])
        axes[1, 0].set_title('Loan Purpose vs. Default Risk')
        axes[1, 0].tick_params(axis='x', rotation=45)

        sns.boxplot(x='TARGET', y='AMT_ANNUITY_CURR', data=df, ax=axes[1, 1])
        axes[1, 1].set_title('Loan Annuity vs. Default Risk')

        st.pyplot(fig)

        # Difference in approval rates between different types of loans (cash vs. revolving)
        st.header(":violet[Approval Rates Between Cash and Revolving Loans:]")
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.countplot(x='NAME_CONTRACT_TYPE_CURR', hue='TARGET', data=df, ax=ax)
        ax.set_title('Loan Type vs. Approval/Default Counts')

        st.pyplot(fig)

        # Previous application outcomes affecting future default risk
        st.header(":violet[Previous Application Outcomes vs. Future Default Risk:]")
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.countplot(x='TARGET', hue='NAME_CONTRACT_STATUS', data=df, ax=ax)
        ax.set_title('Previous Application Outcomes vs. Future Default Risk')

        st.pyplot(fig)
