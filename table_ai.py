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
def get_reply(prompt_template, parameter, model):
    prompt = prompt_template.replace("{parameter}", parameter)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
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
    # Token cost for the models
    token_cost = {
        "gpt-4o": {"input": 5.00, "output": 15.00},  # per 1M tokens
        "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "chatgpt-4o-latest": {"input": 5.00, "output": 15.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    }
    
    # Estimate the number of tokens based on text length
    total_characters = sum(len(str(df.iloc[i, col])) for col in range(2, df.shape[1]) for i in range(2, df.shape[0]))
    estimated_input_tokens = total_characters / 4  # Approximate: 4 characters per token
    
    # Estimate the cost based on input tokens (output tokens are typically similar in size to input tokens)
    cost_input = (estimated_input_tokens / 1_000_000) * token_cost[model]["input"]
    cost_output = (estimated_input_tokens / 1_000_000) * token_cost[model]["output"]
    
    total_cost = cost_input + cost_output

    return total_cost

# Streamlit UI components
st.title('GPT Excel Processor')
st.write("Upload an Excel file with prompts and parameters.")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file and client.api_key:
    # Load the Excel file immediately after it's uploaded
    df, error = load_file(uploaded_file)
    
    if error:
        st.error(error)
    else:
        # GPT Model selection
        model = st.selectbox("Choose GPT Model", options=[
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
            "chatgpt-4o-latest",
            "gpt-4-turbo"
        ])

        # Estimate cost
        estimated_cost = estimate_cost(df, model)
        st.write(f"Estimated Cost for processing with {model}: ${estimated_cost:.4f}")
        
        # Debug checkbox
        debug = st.checkbox("Show Debug Info")
        
        # Only show the DataFrame if Debug is checked
        if debug:
            st.write("DataFrame loaded successfully:")
            st.dataframe(df)
        
        # Run button
        if st.button("Run"):
            st.write("Processing Data...")
            
            # Define the starting row for parameters and columns for prompts
            parameters_start_row = 2
            prompts_row = 1
            
            # Recalculate the total cells for progress
            total_cells = (df.shape[0] - parameters_start_row) * (df.shape[1] - 2)
            progress_bar = st.progress(0)
            processed_cells = 0

            # Iterate over each row starting from the row where parameters start
            for i in range(parameters_start_row, df.shape[0]):
                parameter = df.iloc[i, 1]  # Get the parameter from the second column (Column B)

                # Debugging: Log parameter value if Debug is checked
                if debug:
                    st.write(f"Processing Row {i + 1}: Parameter = {parameter}")
                
                # Iterate over each column starting from the third column (the prompts)
                for col in range(2, df.shape[1]):
                    prompt_template = df.iloc[prompts_row, col]  # Get the prompt template from the second row
                    
                    # Debugging: Log prompt value if Debug is checked
                    if debug:
                        st.write(f"Processing Column {col + 1}: Prompt = {prompt_template}")  
                    
                    if prompt_template and parameter:
                        result = get_reply(prompt_template, parameter, model)  # Generate the prompt and get the GPT response
                        
                        # Debugging: Log generated prompt and response if Debug is checked
                        if debug:
                            st.write(f"Generated Prompt: {prompt_template.replace('{parameter}', parameter)}")  
                            st.write(f"GPT Response: {result}")  
                        
                        df.at[i, col] = result  # Store the result back in the DataFrame

                        # Update progress bar
                        processed_cells += 1
                        progress_bar.progress(processed_cells / total_cells)

            # Only show the Final DataFrame if Debug is checked
            if debug:
                st.write("Final DataFrame:")
                st.dataframe(df)
            
            # Create a downloadable Excel file with the results
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
