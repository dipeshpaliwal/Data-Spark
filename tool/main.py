
import streamlit as st
from streamlit_navigation_bar import st_navbar
from tabs.home import home_page
from tabs.visualization import visualization_page
from tabs.prediction import prediction_page
from tabs.about import about_page
import pandas as pd

page = st_navbar(["Home", "Visualization", "Prediction", "About"])


def load_data(file):
    if file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or file.type == "application/vnd.ms-excel":
        data = pd.read_excel(file, sheet_name=None)  # Load all sheets
    elif file.type == "text/csv":
        data = pd.read_csv(file)
    elif file.type == "application/json":
        data = pd.read_json(file)
    return data
# Initialize session_state
if 'data' not in st.session_state:
    st.session_state.data = None  

# Mapping of pages to functions
page_functions = {
    "Home": home_page,
    "Visualization": visualization_page,
    "Prediction": prediction_page,
    "About": about_page
}

# Main function to create the navigation
def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Get the function based on the current page
    page_function = page_functions.get(page, home_page)
    
    # Call the function
    page_function()








            

        



def display():

    if page == 'Home':
        home_page()
    elif page == 'Visualization':
        visualization_page()
    elif page == 'Prediction':
        prediction_page()
    elif page == 'About':
        about_page()
display()
