import streamlit as st
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables (API keys, etc.)
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to get a reply from OpenAI
def get_reply(prompt_template, model, system_context):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_context},
                {"role": "user", "content": prompt_template},
            ],
            model=model,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Function to load an Excel file
def load_file(uploaded_file):
    try:
        return pd.read_excel(uploaded_file, header=None, dtype=str, engine='openpyxl'), None
    except Exception as e:
        return None, f"Error loading Excel file: {e}"

# Function to refine system context using initial parameters and prompts
def refine_context(initial_context, parameters, prompts):
    combined_input = (
        f"Here is the basic system context:\n\n{initial_context}\n\n"
        f"Now, refine this context based on the following example data:\n\n"
        f"Parameters: {parameters}\nPrompts: {prompts}\n\n"
        f"Please return only the refined context without any other response."
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": initial_context},
                {"role": "user", "content": combined_input},
            ],
            model="chatgpt-4o-latest",  # Always use this model for refining context
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error refining context: {str(e)}"

# Streamlit app configuration and title
st.set_page_config(page_title='TableGen AI', page_icon='🔗')
st.title('🔗 TableGen AI')
st.write("Upload an Excel file with prompts and parameters.")

# Initialize session state variables
st.session_state.setdefault('refined_context', "")
st.session_state.setdefault('last_uploaded_file', None)
st.session_state.setdefault('context_area', "")

# File uploader widget
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

# GPT model selection widget
model = st.selectbox("Choose GPT Model", options=[
    "gpt-4o",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
    "chatgpt-4o-latest",
    "gpt-4-turbo"
], index=2)

# Process the uploaded file and refine context if new file uploaded
if uploaded_file and client.api_key:
    df, error = load_file(uploaded_file)

    if error:
        st.error(error)
    else:
        if uploaded_file != st.session_state.last_uploaded_file:
            st.session_state.last_uploaded_file = uploaded_file

            # Analyze the first 3 parameters and define the context
            first_three_parameters = [df.iloc[i, 1] for i in range(2, min(5, df.shape[0]))]
            sample_prompts = [df.iloc[1, col] for col in range(2, min(5, df.shape[1]))]

            # Refine the context using the first three parameters and prompts
            default_context = (
                "You are working with an Excel file where each reply is one cell. "
                "Provide concise and short replies suitable for an Excel cell. "
                "For numbers, only return numbers; for addresses, only the address. "
                "If unsure, leave the cell empty or use 'unknown'."
            )
            st.session_state.refined_context = refine_context(default_context, first_three_parameters, sample_prompts)
            st.session_state.context_area = st.session_state.refined_context  # Update context for display

# Display the text area widget with the current context
context_prompt = st.text_area("Context for OpenAI API", value=st.session_state.context_area, height=200, key="context_area")

# Debug checkbox widget
debug = st.checkbox("Show Debug Info")

# Button to trigger processing
if st.button("Run"):
    # Use the latest context prompt from the text area
    context_prompt = st.session_state["context_area"]

    st.write("Processing Data...")

    # Dynamically set starting rows and columns based on your DataFrame structure
    parameters_start_row = 2  # Data starts from this row
    prompts_row = 1  # Prompts are defined in this row

    # Identify the first prompt column dynamically
    first_prompt_col = 1
    while first_prompt_col < df.shape[1] and pd.isna(df.iloc[prompts_row, first_prompt_col]):
        first_prompt_col += 1

    # Initialize progress tracking variables
    processed_cells = 0
    total_cells = (df.shape[0] - parameters_start_row) * (df.shape[1] - first_prompt_col)
    progress_bar = st.progress(0)

    # Iterate over each row and column, process data with GPT model
    for i in range(parameters_start_row, df.shape[0]):
        if debug:
            st.write("")
            st.write("")
            st.write(f"**Processing Row {i + 1}**") 

        for col in range(first_prompt_col, df.shape[1]):
            prompt_template = df.iloc[prompts_row, col]

            if pd.notna(prompt_template):  # Ensure the prompt_template is not NaN
                modified_prompt = prompt_template

                # Iterate over all parameter columns dynamically
                for param_col in range(1, first_prompt_col):  # parameter columns are before first_prompt_col
                    column_name = df.iloc[0, param_col]  # Get the column name from the header row
                    placeholder = f"{{{column_name}}}"

                    if placeholder in modified_prompt:
                        # Replace placeholder with actual value from the current row
                        column_value = str(df.iloc[i, param_col])
                        modified_prompt = modified_prompt.replace(placeholder, column_value)

                        if debug:
                            st.write(f"Replaced {placeholder} with {column_value} in Prompt {col + 1}")

                # Check if any replacement was done
                if modified_prompt != prompt_template:
                    if debug:
                        st.write(f"Processed Prompt in Column {col + 1}: {modified_prompt}")

                    # Get GPT response
                    result = get_reply(modified_prompt, model, context_prompt)

                    if debug:
                        st.write(f"GPT Response: {result}")

                    # Store the result in the DataFrame
                    df.at[i, col] = result

                processed_cells += 1  # Increment the processed cells count
                progress_bar.progress(processed_cells / total_cells)

    # Complete the progress bar
    progress_bar.progress(100)

    if debug:
        st.write(f"Context: {context_prompt}")
        st.write("Final DataFrame:")
        st.dataframe(df)

    # Allow user to download the processed file
    output = BytesIO()
    df.to_excel(output, index=False, header=False)
    output.seek(0)

    st.success("Processing complete!")
    st.download_button(
        label="Download processed file",
        data=output,
        file_name="processed_prompts.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
