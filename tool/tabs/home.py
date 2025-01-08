import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.subplots as sp
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce


import streamlit as st

import streamlit as st

def home_page():
    # Inject custom CSS for styling, background image, and full-width layout
    st.markdown(
        """
        <style>
         {
            background-color: white; /* Set background color to white */
            color: #333333; /* Dark text for better contrast on white background */
            font-family: 'Roboto', sans-serif; /* Set font */
            margin: 0;
            padding: 0;
            width: 100%;
        }

        /* Main Heading */
        .main-heading {
            font-family: 'Futura', sans-serif;
            font-weight: bold;
            font-size: 75px;
            text-align: center;
            color: #00FFFF; /* Bright Cyan for Data */
            margin-top: 40px;
            text-shadow: 3px 3px 10px rgba(0, 255, 255, 0.4); /* Cyan glow effect */
        }
        .spark {
            font-family: 'Montserrat', sans-serif;
            font-weight: bold;
            margin-left: 20px;
            color: #FF00FF; /* Magenta for Spark */
            font-size: 85px;
            text-shadow: 2px 2px 8px rgba(255, 0, 255, 0.4); /* Magenta glow effect */
        }



        /* Subheading */
        .subheading {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 40px;
            color: #FF6600; 
            


            letter-spacing: 2px; /* Increase spacing for a modern look */
            
        }


        /* Layout container for text and image */
        .intro-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: flex-start;
            margin: 40px auto;
            max-width: 95%;
            padding: 30px;
            background: rgba(34, 34, 34, 0.8); /* Dark background with transparency */
            border-radius: 10px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2); /* Subtle shadow */
        }

        /* Text section styling */
        .intro-text {
            font-size: 20px;
            color: #E0E0E0;
            flex: 1;
            margin-right: 30px;
            min-width: 300px;
            max-width: 60%;
            line-height: 1.6;
        }

        ul {
            margin: 10px 0;
            padding-left: 20px;
            color: #B0BEC5;
        }

        li {
            margin-bottom: 10px;
        }

        /* Image container styling */
        .image-container {
            text-align: center;
            width: 35%;
            flex-shrink: 0;
            transition: transform 0.3s ease; /* Smooth zoom effect */
        }

        img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.3); /* Add depth */
        }

        /* Hover zoom effect on image container */
        .image-container:hover {
            transform: scale(1.1); /* Zoom in on hover */
        }

        /* Feature Cards */
        .feature-card {
            background-color: #333333;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            margin: 20px 0;
            flex: 1;
            min-width: 280px;
            transition: transform 0.3s ease; /* Smooth zoom effect */
        }

        .feature-card-header {
            font-size: 24px;
            font-weight: bold;
            color: #FF6600;
            margin-bottom: 15px;
        }

        .feature-card-content {
            font-size: 16px;
            color: #E0E0E0;
        }

        /* Hover zoom effect on feature card */
        .feature-card:hover {
            transform: scale(1.05); /* Zoom in on hover */
        }

        /* Responsive Layout */
        @media (max-width: 768px) {
            .intro-container {
                flex-direction: column;
                align-items: center;
            }

            .intro-text {
                margin-right: 0;
                text-align: center;
            }

            .image-container {
                margin-top: 20px;
                width: 80%;
            }

            .feature-card {
                margin: 20px auto;
                width: 90%;
            }
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    # Main heading with custom animation
    st.markdown(
        """
        <div class="main-heading">
            Data<span class="spark">Spark</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Subheading and intro section with text and image side by side, text aligned left
    st.markdown(
        """
        <div class="subheading">
            Discover the Power of Your Data
        </div>
        <div class="intro-container">
            <div class="intro-text">
                Welcome to <strong>Data Spark</strong>, a powerful web app designed to <strong>empower your data journey</strong>. Our platform offers robust tools for <strong>data analytics</strong> and <strong>visualization</strong>, allowing you to uncover insights and make informed decisions. Key features include:
                <ul>
                    <li><strong>Data Cleaning:</strong> Efficiently clean and preprocess your data to ensure accuracy and consistency.</li>
                    <li><strong>Visualization:</strong> Create stunning and interactive visualizations to better understand your data.</li>
                    <li><strong>Machine Learning:</strong> Apply advanced machine learning algorithms to predict trends and outcomes.</li>
                </ul>
                Dive in and explore the true potential of your data!
            </div>
            <div class="image-container">
                <img src="https://img.freepik.com/free-vector/site-stats-concept-illustration_114360-1434.jpg?ga=GA1.1.648290210.1723099833&semt=ais_hybrid" alt="Data visualization illustration"/>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature Cards - Highlight the platform's key features in cards
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-card-header">Data Cleaning</div>
            <div class="feature-card-content">
                Efficiently clean and preprocess your data for accuracy, eliminating errors and ensuring quality before analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-card-header">Data Visualization</div>
            <div class="feature-card-content">
                Create stunning and interactive charts, graphs, and dashboards to visualize your data in an engaging way.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-card-header">Machine Learning</div>
            <div class="feature-card-content">
                Use machine learning models to analyze trends and predict future outcomes based on your data.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Call the home_page function to display the content
home_page()




