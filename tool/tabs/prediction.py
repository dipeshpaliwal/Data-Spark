import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import category_encoders as ce
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Linear Regression": LinearRegression(),
    "Decision Tree (Regressor)": DecisionTreeRegressor(),
    "Support Vector Machine (Regressor)": SVR(),
    "Random Forest (Regressor)": RandomForestRegressor()
}
model_names = list(models.keys())
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx") or file.name.endswith(".xls"):
        return pd.read_excel(file, sheet_name=None)
    elif file.name.endswith(".json"):
        return pd.read_json(file)
    else:
        return None

def apply_default_filters(data):
    """ Apply default filters: remove nulls, encode categorical values, remove duplicates, and standardize numeric data. """
    data_cleaned = data.copy()
    
    # Remove rows with any null values
    data_cleaned = data_cleaned.dropna()
    
    # Convert categorical columns using Label Encoding
    categorical_columns = data_cleaned.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in categorical_columns:
        data_cleaned[col] = le.fit_transform(data_cleaned[col])
    
    # Remove duplicate rows
    data_cleaned = data_cleaned.drop_duplicates()
    
    # Apply Standard Scaling to numeric columns
    
    
    return data_cleaned

def scale_data(data, scaling_method):
    """ Scale data based on the selected scaling method. """
    scaler = None
    if scaling_method == "Standard Scaling":
        scaler = StandardScaler()
    elif scaling_method == "Min-Max Scaling":
        scaler = MinMaxScaler()
    
    if scaler:
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    return data

def train_models(X_train, y_train, X_test, y_test, problem_type):
    results = []
    models = {}
    
    if problem_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Support Vector Machine": SVC(),
            "Random Forest": RandomForestClassifier()
        }
    elif problem_type == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree (Regressor)": DecisionTreeRegressor(),
            "Support Vector Machine (Regressor)": SVR(),
            "Random Forest (Regressor)": RandomForestRegressor()
        }
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if problem_type == "Regression":
            metric = mean_squared_error(y_test, y_pred)
        elif problem_type == "Classification":
            metric = accuracy_score(y_test, y_pred)
        
        results.append({"Model": model_name, "Metric": metric})
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        if problem_type == "Regression":
            best_model_name = results_df.loc[results_df['Metric'].idxmin()]['Model']
        elif problem_type == "Classification":
            best_model_name = results_df.loc[results_df['Metric'].idxmax()]['Model']
    else:
        best_model_name = None
    
    return results_df, best_model_name
def convert_to_float(value):
    """ Convert a value to float if possible. """
    try:
        return float(value)
    except ValueError:
        return value 

def prediction_page():
    st.title("Prediction Page")
    st.markdown('<style>h1 { color: #FF0800; }</style>', unsafe_allow_html=True)
    image_url = r"C:\\Users\\91720\\Desktop\\python env\\exploration.png"
    st.image(image_url, width=100)
    st.markdown('<h2 style="color:#00FF00; text-align: center;">Data Explore</h2>', unsafe_allow_html=True)
    st.write("Upload a file to explore data:")

    # Initialize session state variables if not already set
    if "data_prediction" not in st.session_state:
        st.session_state.data_prediction = None
    if "selected_sheet" not in st.session_state:
        st.session_state.selected_sheet = None
    if "data_cleaned" not in st.session_state:
        st.session_state.data_cleaned = pd.DataFrame()
    if "display_data" not in st.session_state:
        st.session_state.display_data = None
    if "default_filtered_data" not in st.session_state:
        st.session_state.default_filtered_data = None
    if "current_option" not in st.session_state:
        st.session_state.current_option = "None"
    if "filters_applied" not in st.session_state:
        st.session_state.filters_applied = False
    if "model_results" not in st.session_state:
        st.session_state.model_results = pd.DataFrame()
    if "training_data" not in st.session_state:
        st.session_state.training_data = None
    if "target_column" not in st.session_state:
        st.session_state.target_column = None
    if "model_type" not in st.session_state:
        st.session_state.model_type = None
    if "problem_type" not in st.session_state:
        st.session_state.problem_type = "None"
    if "scaling_applied" not in st.session_state:
        st.session_state.scaling_applied = False
    if "scaling_method" not in st.session_state:
        st.session_state.scaling_method = "None"
    if "best_model" not in st.session_state:
        st.session_state.best_model = None
    if "X_train" not in st.session_state:
        st.session_state.X_train = None
    if "X_test" not in st.session_state:
        st.session_state.X_test = None
    if "y_train" not in st.session_state:
        st.session_state.y_train = None
    if "y_test" not in st.session_state:
        st.session_state.y_test = None
    if "model" not in st.session_state:
        st.session_state.model = None

    file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "json"])

    if file is not None:
        st.session_state.data_prediction = load_data(file)
        if isinstance(st.session_state.data_prediction, dict):
            st.session_state.selected_sheet = list(st.session_state.data_prediction.keys())[0]

    data = st.session_state.data_prediction

    if data is not None:
        st.write("Preview of the uploaded data:")
        st.write(data)

        if isinstance(data, dict):
            sheet_names = list(data.keys())
            selected_sheet = st.selectbox(
                "Select sheet",
                sheet_names,
                index=sheet_names.index(st.session_state.selected_sheet) if st.session_state.selected_sheet in sheet_names else 0
            )
            st.session_state.selected_sheet = selected_sheet
            data_selected = data[selected_sheet]
            st.write("Preview of selected sheet:")
            st.write(data_selected)
        else:
            data_selected = data

        # Apply default filters and set default_filtered_data
        if st.session_state.default_filtered_data is None:
            st.session_state.default_filtered_data = apply_default_filters(data_selected)

        st.session_state.data_cleaned = data_selected.copy()
        data_cleaned = st.session_state.data_cleaned
        numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns

        st.sidebar.title("Data Options")
        data_option = st.sidebar.selectbox(
            "Select option",
            ["None", "Exploratory Data Analysis", "Data Cleaning and Preprocessing", "Training and Prediction"],
            index=["None", "Exploratory Data Analysis", "Data Cleaning and Preprocessing", "Training and Prediction"].index(st.session_state.current_option) if st.session_state.current_option in ["None", "Exploratory Data Analysis", "Data Cleaning and Preprocessing", "Training and Prediction"] else 0
        )
        st.session_state.current_option = data_option

        if data_option == "Exploratory Data Analysis":
            st.write("### Exploratory Data Analysis Options:")
            show_head = st.checkbox("Show Head")
            if show_head:
                st.write("#### Head of the Data:")
                st.write(data_selected.head())
            show_tail = st.checkbox("Show Tail")
            if show_tail:
                st.write("#### Tail of the Data:")
                st.write(data_selected.tail())
            show_duplicates = st.checkbox("Show Duplicate Records")
            if show_duplicates:
                st.write("#### Duplicate Records:")
                st.write(data_selected[data_selected.duplicated()])
            show_null_values = st.checkbox("Show Null Values")
            if show_null_values:
                st.write("#### Null Values:")
                st.write(data_selected.isnull().sum())
            show_data_types = st.checkbox("Show Data Types")
            if show_data_types:
                st.write("#### Data Types:")
                st.write(data_selected.dtypes)
            show_description = st.checkbox("Show Description")
            if show_description:
                st.write("### Data Description:")
                st.write(data_cleaned.describe(include='all'))
            show_shape = st.checkbox("Show Shape")
            if show_shape:
                st.write("### Data Shape:")
                st.write(f"The dataset has {data_cleaned.shape[0]} rows and {data_cleaned.shape[1]} columns.")
            show_unique = st.checkbox("Show Unique Values")
            if show_unique:
                st.write("### Unique Values Summary")
                unique_summary = pd.DataFrame({
                    'Column': data_cleaned.columns,
                    'Unique Values': [data_cleaned[col].nunique() for col in data_cleaned.columns]
                })
                st.write(unique_summary)

        elif data_option == "Data Cleaning and Preprocessing":
            st.write("### Data Cleaning Options:")
            encode_categorical = st.checkbox("Convert Categorical Columns", value=False)
            handle_nulls = st.checkbox("Handle Null Values", value=False)
            remove_duplicates = st.checkbox("Remove Duplicates", value=False)
            handle_garbage_values = st.checkbox("Handle Garbage Values", value=False)
            apply_scaling = st.checkbox("Apply Scaling", value=False)

            if encode_categorical:
                st.write("#### Encoding Options:")
                categorical_columns = data_cleaned.select_dtypes(include=['object']).columns.tolist()
                if categorical_columns:
                    columns_to_encode = st.multiselect("Select columns to encode", categorical_columns)
                    encoding_method = st.selectbox("Select encoding method", ["Binary Encoding", "Label Encoding"])

                    if columns_to_encode:
                        if encoding_method == "Label Encoding":
                            le = LabelEncoder()
                            for col in columns_to_encode:
                                if col in data_cleaned.columns:
                                    data_cleaned[col] = le.fit_transform(data_cleaned[col])
                        elif encoding_method == "Binary Encoding":
                            binary_encoder = ce.BinaryEncoder(cols=columns_to_encode)
                            data_cleaned = binary_encoder.fit_transform(data_cleaned)
                    st.session_state.filters_applied = True
                else:
                    st.write("No categorical columns found.")

            if handle_nulls:
                st.write("#### Null Handling Options:")
                remove_nulls = st.checkbox("Remove Rows with Null Values", value=False)
                replace_nulls = st.checkbox("Replace Null Values", value=False)
                
                if replace_nulls:
                    null_columns = data_cleaned.columns[data_cleaned.isnull().any()].tolist()
                    if null_columns:
                        column_to_fill = st.selectbox("Select column to replace null values", null_columns)
                        replacement_option = st.radio(
                            "Select replacement method",
                            ("Mean", "Median", "Mode", "Constant Value")
                        )
                        if replacement_option == "Mean":
                            replacement_value = data_cleaned[column_to_fill].mean()
                        elif replacement_option == "Median":
                            replacement_value = data_cleaned[column_to_fill].median()
                        elif replacement_option == "Mode":
                            replacement_value = data_cleaned[column_to_fill].mode()[0]
                        elif replacement_option == "Constant Value":
                            replacement_value = st.text_input("Enter constant value")
                        
                        if replacement_value is not None:
                            data_cleaned[column_to_fill].fillna(replacement_value, inplace=True)

                if remove_nulls:
                    data_cleaned = data_cleaned.dropna()
                
                st.session_state.filters_applied = True

            if remove_duplicates:
                data_cleaned = data_cleaned.drop_duplicates()
                st.session_state.filters_applied = True

            if handle_garbage_values:
                st.write("#### Handling Garbage Values:")
                st.write("Remove special characters from columns")
                special_char_pattern = r'[^A-Za-z0-9\s]+'
                for column in data_cleaned.columns:
                    if data_cleaned[column].dtype == 'object':
                        data_cleaned = data_cleaned[~data_cleaned[column].str.contains(special_char_pattern, regex=True)]
                st.session_state.filters_applied = True

            if apply_scaling:
                st.write("#### Scaling Options:")
                scaling_method = st.selectbox("Select scaling method", ["None", "Standard Scaling", "Min-Max Scaling"])
                st.session_state.scaling_method = scaling_method

                if scaling_method != "None":
                    data_cleaned = scale_data(data_cleaned, scaling_method)
                    st.session_state.scaling_applied = True

            if st.session_state.filters_applied or st.session_state.scaling_applied:
                st.write("#### Cleaned and Scaled Data:")
                st.write(data_cleaned)
                st.session_state.default_filtered_data = data_cleaned.copy()
                st.session_state.display_data = data_cleaned.copy()
            else:
                st.write("#### Default Filtered Data:")
                st.write(st.session_state.default_filtered_data)
                st.session_state.display_data = st.session_state.default_filtered_data




 

        elif data_option == "Training and Prediction":
            st.write("### Model Training and Prediction")
            
            if st.session_state.data_cleaned.empty:
                st.warning("Please upload and clean data before proceeding.")
                return

            # Use default filtered data if no additional filtering was applied
            data_to_use = st.session_state.display_data if (st.session_state.filters_applied or st.session_state.scaling_applied) else st.session_state.default_filtered_data
            
            target_column = st.selectbox("Select target column", options=data_to_use.columns)
            st.session_state.target_column = target_column
            problem_type = st.selectbox("Select problem type", ["Classification", "Regression"])
            st.session_state.problem_type = problem_type

            X = data_to_use.drop(columns=[target_column])
            y = data_to_use[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            model_results, best_model_name = train_models(X_train, y_train, X_test, y_test, problem_type)
            st.session_state.model_results = model_results
            st.session_state.best_model_name = best_model_name

            st.write("### Model Performance:")
            st.write(model_results)
            st.write(f"**Best Model:** {best_model_name}")

            st.write("### Select a Model to Train and Predict:")
            model_names = list(models.keys())
            selected_model_name = st.selectbox("Select Model", model_names)
            st.session_state.model = models[selected_model_name]

            if st.button("Train Model"):
                try:
                    st.session_state.model.fit(X_train, y_train)
                    st.write(f"**Model {selected_model_name} trained successfully!**")
                except Exception as e:
                    st.error(f"Error training model: {e}")

            st.write("### Select Features for Prediction:")
            all_features = X.columns.tolist()
            selected_features = st.multiselect("Select columns for prediction", options=all_features)

            if not selected_features:
                st.warning("Please select at least one feature for prediction.")
                return
            
            st.write("### Enter Values for Prediction:")
            user_input = {}
            
            for feature in selected_features:
                feature_type = str(data_to_use[feature].dtype)  # Get the data type of the feature
                if "int" in feature_type or "float" in feature_type:
                    user_input[feature] = st.number_input(f"Enter value for {feature}", format="%.2f", key=f"numberinput_{feature}")
                elif "object" in feature_type or "str" in feature_type:
                    user_input[feature] = st.text_input(f"Enter value for {feature}", key=f"textinput_{feature}")

            if st.button("Predict"):
                input_df = pd.DataFrame([user_input], columns=selected_features)

                # Handle missing values
                imputer = SimpleImputer(strategy='mean')
                input_df = imputer.fit_transform(input_df)

                # Ensure the correct shape of input_df
                if st.session_state.scaling_applied:
                    scaler = scale_data(X, st.session_state.scaling_method)
                    input_df = scaler.transform(input_df)

                model = st.session_state.model
                
                try:
                    prediction = model.predict(input_df)
                    st.write(f"**Prediction Result:** {prediction[0]}")
                except NotFittedError:
                    st.error(f"Error: The selected model '{selected_model_name}' is not fitted. Please train the model first.")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")



def convert_to_float(value):
    """Attempt to convert a string value to a float. Return the original value if conversion fails."""
    try:
        return float(value)
    except ValueError:
        return value
