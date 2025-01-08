import pandas as pd
import streamlit as st
from streamlit_navigation_bar import st_navbar
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.subplots as sp
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

import streamlit as st

def about_page():
    st.title("About Page")
    st.markdown('<style>h1 { color: #FF0800; }</style>', unsafe_allow_html=True)
    
    # About the App
    st.write("""
    This app is designed to help with **data analysis, prediction, and visualization**. It allows users to upload datasets, 
    perform exploratory data analysis (EDA), preprocess the data, and build machine learning models for making predictions. 
    The app uses various powerful libraries like **Pandas, Scikit-learn, Plotly, and Matplotlib** to analyze and visualize data.
    
    - **Data Analysis**: Load and explore datasets.
    - **Prediction**: Use machine learning algorithms to make predictions on the data.
    - **Visualization**: Generate interactive plots and graphs for better understanding of the data.
    """)

    # Contact Information with Icons
    st.write("### Contact Me")
    st.write("For any inquiries or suggestions, feel free to reach out:")

    # Create a row for the icons
    contact_icons = """
    <div style="display: flex; align-items: center;">
        <a href="mailto:paliwaldipesh336@gmail.com" style="margin-right: 20px;">
            <img src="https://img.icons8.com/?size=48&id=qyRpAggnV0zH&format=png" width="40" height="40"/>
        </a>
        <a href="https://www.linkedin.com/in/dipesh-paliwal-351b22248/">
            <img src="https://img.icons8.com/?size=48&id=xuvGCOXi8Wyg&format=png" width="40" height="40"/>
        </a>
    </div>
    """

    st.markdown(contact_icons, unsafe_allow_html=True)

# You can call about_page() function in your app wherever you need the About page to appear
