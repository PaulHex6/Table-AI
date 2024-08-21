import streamlit as st
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import os
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

# Load environment variables (API keys, etc.)
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_reply(prompt_template, model, system_context):
    """Get a reply from the OpenAI model."""
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

def load_file(uploaded_file):
    """Load an Excel file and return a DataFrame."""
    try:
        return pd.read_excel(uploaded_file, header=None, dtype=str, engine='openpyxl'), None
    except Exception as e:
        return None, f"Error loading Excel file: {e}"

def refine_context(initial_context, parameters, prompts):
    """Refine system context using initial parameters and prompts."""
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
            model="chatgpt-4o-latest",
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error refining context: {str(e)}"

def extract_and_refine_context(df):
    """Extract the first three parameters and refine the context."""
    first_three_parameters = [df.iloc[i, 1] for i in range(2, min(5, df.shape[0]))]
    sample_prompts = [df.iloc[1, col] for col in range(2, min(5, df.shape[1]))]

    default_context = (
        "You are working with an Excel file where each reply is one cell. "
        "Provide concise and short replies suitable for an Excel cell. "
        "For numbers, only return numbers; for addresses, only the address. "
        "If unsure, leave the cell empty or use 'unknown'."
    )
    return refine_context(default_context, first_three_parameters, sample_prompts)

def process_excel_file(df, model, context_prompt, debug):
    """Process the Excel file and interact with the GPT model."""
    parameters_start_row = 2
    prompts_row = 1
    first_prompt_col = find_first_prompt_column(df, prompts_row)

    processed_cells = 0
    total_cells = (df.shape[0] - parameters_start_row) * (df.shape[1] - first_prompt_col)
    progress_bar = st.progress(0)

    for i in range(parameters_start_row, df.shape[0]):
        if debug:
            st.write("") 
            st.write(f"**Processing Row {i + 1}**") 

        for col in range(first_prompt_col, df.shape[1]):
            prompt_template = df.iloc[prompts_row, col]

            if pd.notna(prompt_template):
                modified_prompt = replace_placeholders(prompt_template, df, i, first_prompt_col, debug)
                result = get_reply(modified_prompt, model, context_prompt)

                if debug:
                    st.write(f"GPT Response: {result}")

                df.at[i, col] = result
                processed_cells += 1
                progress_bar.progress(processed_cells / total_cells)

    progress_bar.progress(100)
    return df

def find_first_prompt_column(df, prompts_row):
    """Identify the first prompt column dynamically."""
    first_prompt_col = 1
    while first_prompt_col < df.shape[1] and pd.isna(df.iloc[prompts_row, first_prompt_col]):
        first_prompt_col += 1
    return first_prompt_col

# Dictionary to cache the results of fetched URLs
url_cache = {}

def fetch_unformatted_text(url):
    """Fetches the unformatted text content from a given URL. IMPORTANT: must start with http:// or https://"""

    if url in url_cache:
        return url_cache[url]

    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors

        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract all the text without any formatting
        text = soup.get_text()
        return text.strip()  # Return clean text
    except Exception as e:
        return f"Error fetching text from URL: {str(e)}"

def replace_placeholders(prompt_template, df, row_idx, first_prompt_col, debug):
    """Replace placeholders in the prompt template with actual values."""
    modified_prompt = prompt_template
    for param_col in range(1, first_prompt_col):
        column_name = df.iloc[0, param_col]
        placeholder = f"{{{column_name}}}"
        if placeholder in modified_prompt:
            column_value = str(df.iloc[row_idx, param_col])

            # Check if the column_value is a URL, IMPORTANT: must start with http:// or https://
            if column_value.startswith(('http://', 'https://')):
                column_value = fetch_unformatted_text(column_value)

            modified_prompt = modified_prompt.replace(placeholder, column_value)
            if debug:
                st.write(f"Replaced {placeholder} with {column_value} in Prompt {param_col + 1}")
    return modified_prompt

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title='TableGen AI', page_icon='🔗')
    st.title('🔗 TableGen AI')
    st.write("Upload an Excel file with prompts and parameters.")
    
    st.session_state.setdefault('refined_context', "")
    st.session_state.setdefault('last_uploaded_file', None)
    st.session_state.setdefault('context_area', "")

    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    model = st.selectbox("Choose GPT Model", options=[
        "gpt-4o",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini",
        "chatgpt-4o-latest",
        "gpt-4-turbo"
    ], index=2)

    if uploaded_file and client.api_key:
        df, error = load_file(uploaded_file)

        if error:
            st.error(error)
        else:
            if uploaded_file != st.session_state.last_uploaded_file:
                st.session_state.last_uploaded_file = uploaded_file
                st.session_state.refined_context = extract_and_refine_context(df)
                st.session_state.context_area = st.session_state.refined_context

    #context_prompt = st.text_area("Context for OpenAI API", value=st.session_state.context_area, height=200, key="context_area")
    context_prompt = st.text_area(
        "Context for OpenAI API", 
        value=st.session_state.get('context_area', ""), 
        height=200, 
        key="context_area"
    )

    debug = st.checkbox("Show Debug Info")

    if st.button("Run"):
        context_prompt = st.session_state["context_area"]
        st.write("Processing Data...")
        df = process_excel_file(df, model, context_prompt, debug)

        if debug:
            st.write(f"Context: {context_prompt}")
            st.write("Final DataFrame:")
            st.dataframe(df)

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

# Run the main function
if __name__ == "__main__":
    main()
