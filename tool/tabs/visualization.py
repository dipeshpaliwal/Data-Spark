import pandas as pd
import streamlit as st
from streamlit_navigation_bar import st_navbar
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.subplots as sp
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce




def load_data(file):
    if file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or file.type == "application/vnd.ms-excel":
        data = pd.read_excel(file, sheet_name=None)  # Load all sheets
    elif file.type == "text/csv":
        data = pd.read_csv(file)
    elif file.type == "application/json":
        data = pd.read_json(file)
    return data


def visualization_page():
    st.title("Visualization Page")
    st.markdown('<style>h1 { color: #FF0800; }</style>', unsafe_allow_html=True)
    
    image_url = r"C:\Users\91720\Desktop\python env\image1.png"
    st.image(image_url, width=100)
    st.write("Upload a file to visualize data:")

    if 'data_visualization' not in st.session_state:
        st.session_state.data_visualization = None

    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "json"], key='visualization_file')
    
    if file is not None:
        st.session_state.uploaded_file = file
        st.session_state.data_visualization = load_data(file)
        
    data = st.session_state.data_visualization

    if data is not None:
        st.write("Preview of the uploaded data:")
        if isinstance(data, dict):
            sheet_names = list(data.keys())
            selected_sheet = st.selectbox("Select sheet", sheet_names, index=st.session_state.get('selected_sheet_index', 0))
            st.session_state.selected_sheet_index = sheet_names.index(selected_sheet)
            data_selected = data[selected_sheet]
        else:
            data_selected = data

        st.write("Preview of selected sheet:")
        st.write(data_selected)

        # Sidebar for filter options
        st.sidebar.title("Filter Options")
        available_columns = data_selected.columns.tolist()
        selected_columns = st.session_state.get('selected_columns', [])
        
        selected_columns = [col for col in selected_columns if col in available_columns]
        
        selected_columns = st.sidebar.multiselect(
            "Select column(s) to filter",
            options=available_columns,
            default=selected_columns
        )
      
        if selected_columns:
            st.session_state.selected_columns = selected_columns
            filtered_data = data_selected[selected_columns].copy()  # Use copy to avoid modifying original data
            
            for column in selected_columns:
                values_to_filter = st.session_state.get(f'values_to_filter_{column}', [])
                
                # Make sure values_to_filter contains only valid values
                values_to_filter = [val for val in values_to_filter if val in data_selected[column].unique()]

                options = sorted(list(data_selected[column].unique()))
                options.insert(0, "All")  # Add "All" option at the top
                
                selected_values = st.sidebar.multiselect(
                    f"Select values from {column}",
                    options=options,
                    default=values_to_filter
                )
                
                if "All" in selected_values:
                    st.session_state[f'values_to_filter_{column}'] = []
                    # If "All" is selected, no filtering is applied
                    filtered_data = filtered_data
                else:
                    st.session_state[f'values_to_filter_{column}'] = selected_values
                    filtered_data = filtered_data[filtered_data[column].isin(selected_values)]

            st.write("Filtered Data based on selected values:")
            st.write(filtered_data)
            # Graph type selection
            graph_type = st.sidebar.selectbox(
                "Select Graph Type",
                ["None", "Line Plot", "Bar Plot", "Histogram", "Scatter Plot", "Pie Chart", "Horizontal Bar Chart", "Stem Plot", "Combined Plot","Heatmap"],
                index=st.session_state.get('graph_type_index', 0)
            )

            if graph_type != "None":
                st.session_state.graph_type_index = ["None", "Line Plot", "Bar Plot", "Histogram", "Scatter Plot", "Pie Chart", "Horizontal Bar Chart", "Stem Plot", "Combined Plot","Heatmap"].index(graph_type)
                
                x_column = st.sidebar.selectbox(
                    "Select X-axis column",
                    options=filtered_data.columns.tolist(),
                    index=st.session_state.get('x_column_index', 0)
                )
                st.session_state.x_column_index = filtered_data.columns.tolist().index(x_column)

                if graph_type != "Histogram":
                    y_column = st.sidebar.selectbox(
                        "Select Y-axis column",
                        options=filtered_data.columns.tolist(),
                        index=st.session_state.get('y_column_index', 0)
                    )
                    st.session_state.y_column_index = filtered_data.columns.tolist().index(y_column)
                    st.session_state.y_column = y_column
                else:
                    y_column = None

                # Call appropriate plot function
                if graph_type == "Line Plot":
                    plot_line_chart(filtered_data, x_column, y_column)
                elif graph_type == "Bar Plot":
                    plot_bar_chart(filtered_data, x_column, y_column)
                elif graph_type == "Histogram":
                    plot_histogram(filtered_data, x_column)
                elif graph_type == "Scatter Plot":
                    plot_scatter_plot(filtered_data, x_column, y_column)
                elif graph_type == "Pie Chart":
                    plot_pie_chart(filtered_data, x_column, y_column)
                elif graph_type == "Horizontal Bar Chart":
                    plot_horizontal_bar_chart(filtered_data, x_column, y_column)
                elif graph_type == "Stem Plot":
                    plot_stem_plot(filtered_data, x_column, y_column)
                elif graph_type == "Combined Plot":
                    plot_combined_charts(filtered_data, x_column, y_column)
                elif graph_type == "Heatmap":
                    plot_heatmap(filtered_data, x_column)
            else:
                st.session_state.graph_type_index = 0
    else:
        st.write("Please upload a file to visualize data.")

def plot_line_chart(data, x_column, y_column):
    st.subheader("Line Plot")
    fig = px.line(data, x=x_column, y=y_column, title=f'Line Plot: {x_column} vs {y_column}')
    fig.update_layout(
        hovermode='x',
        hoverlabel=dict(bgcolor="black", font_size=12, font_family="Arial"),
        showlegend=True,
        legend=dict(font=dict(family="Arial", size=12, color="black")),
        xaxis=dict(title=x_column, titlefont=dict(size=14, family='Arial', color='black', weight='bold'),
                   tickfont=dict(size=12, family='Arial', color='black', weight='bold')),
        yaxis=dict(title=y_column, titlefont=dict(size=14, family='Arial', color='black', weight='bold'),
                   tickfont=dict(size=12, family='Arial', color='black', weight='bold'))
    )
    fig.update_traces(mode='lines+markers', marker=dict(size=8, color='skyblue', line=dict(width=2, color='DarkSlateGray')))
    st.plotly_chart(fig, use_container_width=True)

def plot_bar_chart(data, x_column, y_column):
    st.subheader("Bar Chart")
    fig = px.bar(data, x=x_column, y=y_column, title=f'Bar Chart: {x_column} vs {y_column}')
    fig.update_layout(
        showlegend=False,
        hovermode='x',
        hoverlabel=dict(bgcolor="black", font_size=12, font_family="Rockwell"),
        xaxis=dict(title=x_column, titlefont=dict(size=14, family='Rockwell', color='black', weight='bold'),
                   tickfont=dict(size=12, family='Rockwell', color='black', weight='bold')),
        yaxis=dict(title=y_column, titlefont=dict(size=14, family='Rockwell', color='black', weight='bold'),
                   tickfont=dict(size=12, family='Rockwell', color='black', weight='bold'))
    )
    fig.update_traces(marker_color='darkgreen', marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

def plot_horizontal_bar_chart(data, x_column, y_column):
    st.subheader("Horizontal Bar Chart")
    fig = px.bar(data, x=y_column, y=x_column, title=f'Horizontal Bar Chart: {x_column} vs {y_column}', orientation='h')
    fig.update_layout(
        showlegend=False,
        hovermode='y',
        hoverlabel=dict(bgcolor="black", font_size=12, font_family="Rockwell"),
        xaxis=dict(title=y_column, titlefont=dict(size=14, family='Rockwell', color='black', weight='bold'),
                   tickfont=dict(size=12, family='Rockwell', color='black', weight='bold')),
        yaxis=dict(title=x_column, titlefont=dict(size=14, family='Rockwell', color='black', weight='bold'),
                   tickfont=dict(size=12, family='Rockwell', color='black', weight='bold'))
    )
    fig.update_traces(marker_color='skyblue', marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

def plot_stem_plot(data, x_column, y_column):
    st.subheader("Stem Plot")
    fig, ax = plt.subplots()
    fig.suptitle(f'Stem Plot: {x_column} vs {y_column}')
    x_values = data[x_column]
    y_values = data[y_column]
    ax.stem(x_values, y_values, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.grid(True)
    st.pyplot(fig)

def plot_scatter_plot(data, x_column, y_column):
    st.subheader("Scatter Plot")
    color_column = st.sidebar.selectbox("Select column for color", data.columns.tolist(), index=1)
    plot_title = f'Scatter Plot: {x_column} vs {y_column}'
    if color_column != "None":
        plot_title += f' colored by {color_column}'
    fig = px.scatter(data, x=x_column, y=y_column, color=color_column, title=plot_title, hover_name=data.index)
    fig.update_traces(hoverinfo='all', hovertemplate='%{hovertext}<extra></extra>')
    fig.update_layout(
        xaxis=dict(title=x_column, titlefont=dict(size=14, family='Arial', color='black', weight='bold'),
                   tickfont=dict(size=12, family='Arial', color='black', weight='bold')),
        yaxis=dict(title=y_column, titlefont=dict(size=14, family='Arial', color='black', weight='bold'),
                   tickfont=dict(size=12, family='Arial', color='black', weight='bold')),
        hovermode='x',
        hoverlabel=dict(bgcolor="black", font_size=12, font_family="Arial"),
        showlegend=True,
        legend=dict(font=dict(family="Arial", size=12, color="black"))
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_pie_chart(data, x_column, y_column):
    st.subheader("Pie Chart")
    fig = px.pie(data, values=y_column, names=x_column, title=f'Pie Chart: {y_column}')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def plot_histogram(data, x_column):
    st.subheader("Histogram")
    num_bins = st.slider("Select number of bins", min_value=5, max_value=50, value=20)
    fig = px.histogram(data, x=x_column, title=f'Histogram of {x_column}', nbins=num_bins)
    fig.update_xaxes(title_text=x_column, title_font=dict(size=14, family='Arial', color='black', weight='bold'),
                     tickfont=dict(size=12, family='Arial', color='black', weight='bold'))
    fig.update_yaxes(title_text="Frequency", title_font=dict(size=14, family='Arial', color='black', weight='bold'),
                     tickfont=dict(size=12, family='Arial', color='black', weight='bold'))
    st.plotly_chart(fig, use_container_width=True)

def plot_combined_charts(data, x_column, y_column):
    # Create a subplot figure with 2 rows and 2 columns
    fig = sp.make_subplots(rows=2, cols=2, 
                           subplot_titles=("Line Plot", "Bar Chart", "Pie Chart", "Scatter Plot"),
                           specs=[[{"type": "xy"}, {"type": "xy"}],
                                  [{"type": "pie"}, {"type": "xy"}]])

    # Line Plot
    line_fig = px.line(data, x=x_column, y=y_column, title=f'Line Plot: {x_column} vs {y_column}')
    for trace in line_fig.data:
        fig.add_trace(trace, row=1, col=1)

    # Bar Chart
    bar_fig = px.bar(data, x=x_column, y=y_column, title=f'Bar Chart: {x_column} vs {y_column}')
    for trace in bar_fig.data:
        fig.add_trace(trace, row=1, col=2)

    # Pie Chart
    pie_fig = px.pie(data, values=y_column, names=x_column, title=f'Pie Chart: {y_column}')
    for trace in pie_fig.data:
        fig.add_trace(trace, row=2, col=1)

    # Scatter Plot
    scatter_fig = px.scatter(data, x=x_column, y=y_column, title=f'Scatter Plot: {x_column} vs {y_column}')
    for trace in scatter_fig.data:
        fig.add_trace(trace, row=2, col=2)

    fig.update_layout(title_text="Combined Plots", title_x=0.5, showlegend=False, height=800)
    st.plotly_chart(fig, use_container_width=True)
def plot_heatmap(data, x_column):
    st.subheader("Heatmap")
    corr = data.corr()
    fig = px.imshow(corr, text_auto=True, title=f'Heatmap of {x_column} Correlations')
    fig.update_layout(
        hovermode='closest',
        hoverlabel=dict(bgcolor="black", font_size=12, font_family="Arial"),
        xaxis=dict(title="Variables", titlefont=dict(size=14, family='Arial', color='black', weight='bold'),
                   tickfont=dict(size=12, family='Arial', color='black', weight='bold')),
        yaxis=dict(title="Variables", titlefont=dict(size=14, family='Arial', color='black', weight='bold'),
                   tickfont=dict(size=12, family='Arial', color='black', weight='bold'))
    )
    st.plotly_chart(fig, use_container_width=True)