import streamlit as st
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client using the new API structure
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Function to get a reply from OpenAI using the correct API method
def get_reply(prompt_template, parameter, model, system_context):
    prompt = prompt_template.replace("{parameter}", parameter)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_context,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Function to load the Excel file
def load_file(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, header=None, dtype=str, engine='openpyxl')  # Force all data to be read as strings
        return df, None
    except Exception as e:
        return None, f"Error loading Excel file: {e}"

# Function to estimate the cost of processing
def estimate_cost(df, model):
    token_cost = {
        "gpt-4o": {"input": 5.00, "output": 15.00},  # per 1M tokens
        "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "chatgpt-4o-latest": {"input": 5.00, "output": 15.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    }
    
    total_characters = sum(len(str(df.iloc[i, col])) for col in range(2, df.shape[1]) for i in range(2, df.shape[0]))
    estimated_input_tokens = total_characters / 4  # Approximate: 4 characters per token
    
    cost_input = (estimated_input_tokens / 1_000_000) * token_cost[model]["input"]
    cost_output = (estimated_input_tokens / 1_000_000) * token_cost[model]["output"]
    
    total_cost = cost_input + cost_output

    return total_cost

# Function to refine system context using initial parameters and prompts
def refine_context(initial_context, parameters, prompts, model):
    combined_input = f"Here is the basic system context:\n\n{initial_context}\n\nNow, refine this context based on the following example data that will be analyzed:\n\nParameters: {parameters}\nPrompts: {prompts}\n\nPlease return only the refined context without any other response."
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": initial_context,
                },
                {
                    "role": "user",
                    "content": combined_input,
                }
            ],
            model="chatgpt-4o-latest",
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error refining context: {str(e)}"

# Streamlit UI components
st.set_page_config(page_title='TableGen AI', page_icon='🔗')
st.title('🔗 TableGen AI')
st.write("Upload an Excel file with prompts and parameters.")

# Initialize session state to store refined context and file identifier
if 'refined_context' not in st.session_state:
    st.session_state.refined_context = None

if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

# GPT Model selection with default as 'gpt-4o-mini'
model = st.selectbox("Choose GPT Model", options=[
    "gpt-4o",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
    "chatgpt-4o-latest",
    "gpt-4-turbo"
], index=2)

# Placeholder for the context text area
context_text_area = st.empty()

# Display the context text area with either the refined context or a placeholder message
if st.session_state.refined_context:
    context_text_area.text_area("Context for OpenAI API", value=st.session_state.refined_context, height=200)
else:
    context_text_area.text_area("Context for OpenAI API", value="", height=200)

if uploaded_file and client.api_key:
    df, error = load_file(uploaded_file)

    if error:
        st.error(error)
    else:
        # Only refine the context if a new file is uploaded
        if uploaded_file != st.session_state.last_uploaded_file:
            st.session_state.last_uploaded_file = uploaded_file

            # Analyze the first 3 parameters and define the context
            first_three_parameters = [df.iloc[i, 1] for i in range(2, min(5, df.shape[0]))]
            sample_prompts = [df.iloc[1, col] for col in range(2, min(5, df.shape[1]))]

            # Refine the context using the first three parameters and prompts
            default_context = "You are working with an Excel file where each reply is one cell. Provide concise and short replies suitable for an Excel cell. For numbers, only return numbers; for addresses, only the address. If unsure, leave the cell empty or use 'unknown'."
            st.session_state.refined_context = refine_context(default_context, first_three_parameters, sample_prompts, model)

            # Update the context area with the refined context
            context_text_area.text_area("Context for OpenAI API", value=st.session_state.refined_context, height=200)

        # Estimate cost
        estimated_cost = estimate_cost(df, model)
        st.write(f"Estimated Cost for processing with {model}: ${estimated_cost:.4f}")

        # Debug checkbox
        debug = st.checkbox("Show Debug Info")

        # Button to trigger processing
        if st.button("Run"):
            st.write("Processing Data...")
            
            parameters_start_row = 2
            prompts_row = 1
            
            total_cells = (df.shape[0] - parameters_start_row) * (df.shape[1] - 2)
            progress_bar = st.progress(0)
            processed_cells = 0

            for i in range(parameters_start_row, df.shape[0]):
                parameter = df.iloc[i, 1]

                if debug:
                    st.write(f"Processing Row {i + 1}: Parameter = {parameter}")
                
                for col in range(2, df.shape[1]):
                    prompt_template = df.iloc[prompts_row, col]

                    if debug:
                        st.write(f"Processing Column {col + 1}: Prompt = {prompt_template}")  
                    
                    if prompt_template and parameter:
                        result = get_reply(prompt_template, parameter, model, st.session_state.refined_context)
                        
                        if debug:
                            st.write(f"Generated Prompt: {prompt_template.replace('{parameter}', parameter)}")  
                            st.write(f"GPT Response: {result}")  
                        
                        df.at[i, col] = result

                        processed_cells += 1
                        progress_bar.progress(processed_cells / total_cells)

            if debug:
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
