# Financial-Risk-Detection

Problem Statement: Develop a predictive model to determine whether a loan applicant is likely to be a defaulter or a repayer. The goal is to help financial institutions identify and manage potential risks associated with loan approvals, ensuring a more informed and stable lending process.​

NAME : RAMYA KRISHNAN A

BATCH: DW75DW76

DOMAIN : DATA SCIENCE

DEMO VIDEO URL :
Linked in URL : www.linkedin.com/in/ramyakrishnan19

# Libraries Used

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    %matplotlib inline
    import seaborn as sns
    import itertools
    import streamlit as st
    from streamlit_option_menu import option_menu

# Libraries Used for Feature Selection

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc , accuracy_score
    from sklearn.metrics import precision_score, recall_score, f1_score
    import pickle

## Menus

### Home

The **Home** menu provides an overview of the project, including:

- Domain: Describe the industry or field your project is related to.
  
- Technology Used: List the main technologies, frameworks, and tools utilized in your project.
  
- Project Overview: Provide a brief summary of the project's main objectives and features.

<img width="1440" alt="Screenshot 2024-03-05 at 11 37 35 AM" src="https://github.com/Ramya19rk/Financial-Risk-Detection/assets/145639838/6674c768-a7f2-4acc-8b82-c3b3e0bb3e4a">
  

### Prediction

The **Prediction** menu is designed for predicting whether a customer is a defaulter or repayer. It includes:

- Form Structure: Describe the structure of the form where users input information.
  
- Submission: Explain the process after the user submits the form and how the prediction is displayed.

 <img width="1440" alt="Screenshot 2024-03-05 at 11 36 50 AM" src="https://github.com/Ramya19rk/Financial-Risk-Detection/assets/145639838/7e81122f-041d-41ac-8dfa-27da1e195ab1">
 

### EDA Analysis

The **EDA Analysis** menu contains five tabs:

<img width="1440" alt="Screenshot 2024-03-05 at 11 37 14 AM" src="https://github.com/Ramya19rk/Financial-Risk-Detection/assets/145639838/e23e2faa-7328-4e98-943a-8ccef2687f80">

 **Univariate Analysis:** Displays univariate insights with options for radio buttons.

<img width="1440" alt="Screenshot 2024-03-05 at 11 37 21 AM" src="https://github.com/Ramya19rk/Financial-Risk-Detection/assets/145639838/7993eb8c-2aad-4cb4-bb7c-396aae791c48">
 
 **Bivariate Bar Analysis:** Presents bivariate analysis using bar charts.

<img width="1440" alt="Screenshot 2024-03-05 at 11 37 35 AM" src="https://github.com/Ramya19rk/Financial-Risk-Detection/assets/145639838/946c1d6e-595b-4d17-acfb-e838b0c0a375">

 
 **Bivariate Relationship Analysis:** Explores all bivariate relationships.

<img width="1440" alt="Screenshot 2024-03-05 at 11 37 43 AM" src="https://github.com/Ramya19rk/Financial-Risk-Detection/assets/145639838/e0ca29f6-342e-4dbd-a8e9-21b76b6d0e34">

 
 **Univariate Merged Analysis:** Shows merged univariate insights.

<img width="1440" alt="Screenshot 2024-03-05 at 11 37 50 AM" src="https://github.com/Ramya19rk/Financial-Risk-Detection/assets/145639838/2d26cf6b-8125-4c59-b937-90979ee48a48">

 
 **Merged Point Plot Analysis:** Visualizes point plot analysis using merged data.
 
<img width="1440" alt="Screenshot 2024-03-05 at 11 38 05 AM" src="https://github.com/Ramya19rk/Financial-Risk-Detection/assets/145639838/a716cf25-cff9-472a-938b-a274f1dc979a">
<img width="1440" alt="Screenshot 2024-03-05 at 11 37 58 AM" src="https://github.com/Ramya19rk/Financial-Risk-Detection/assets/145639838/0df4e676-c529-49e7-9688-b4164ca4aefb">


### Exploration

The **Exploration** menu provides answers to questions in the problem statement document with four tabs:

1. **Table-like Structure:** Displays tabular exploration results.

   <img width="1440" alt="Screenshot 2024-03-05 at 11 38 25 AM" src="https://github.com/Ramya19rk/Financial-Risk-Detection/assets/145639838/76ff9c26-8bbf-4c9a-9a73-e1a43c98dd25">

   
2-4. **Chart Tabs:** Visualizes exploration results using charts.

<img width="1440" alt="Screenshot 2024-03-05 at 11 38 59 AM" src="https://github.com/Ramya19rk/Financial-Risk-Detection/assets/145639838/d84523fc-2898-4dc2-bbe8-cb09b208fa9e">
<img width="1440" alt="Screenshot 2024-03-05 at 11 38 50 AM" src="https://github.com/Ramya19rk/Financial-Risk-Detection/assets/145639838/8b2d8da3-9763-4f2d-b759-df5908102044">
<img width="1440" alt="Screenshot 2024-03-05 at 11 38 39 AM" src="https://github.com/Ramya19rk/Financial-Risk-Detection/assets/145639838/124a8c76-c6c6-4de6-9fbb-b4950c7bbc4c">





