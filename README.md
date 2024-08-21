# 🔗 TableGen AI
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

TableGen AI automatically populates your Excel tables using ChatGPT. Just upload your Excel file, and let the AI fill in the data. Once the process is complete, download the populated table with the generated results.
This is a preliminary version under development, and more updates are coming soon.

## Features

- **Upload Excel Files**: Upload an Excel file with predefined prompts and parameters.
- **Model Selection**: Choose from various GPT models.
- **Dynamic URL Processing**: Automatically fetch URLs for parameters starting with http:// or https://.
- **Debug Information**: Optionally view detailed processing logs for each step.
- **Download Processed File**: After processing, download the updated Excel file containing the GPT responses.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/PaulHex6/Table-AI.git
   cd gpt-excel-processor
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required Python packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API key in a `.env` file**:

   Create a `.env` file in the root directory and add your OpenAI API key:

   ```plaintext
   OPENAI_API_KEY=your-openai-api-key
   ```

## Usage

1. **Run the Streamlit app**:

   ```bash
   streamlit run table_ai.py
   ```

2. **Access the app**:
   
   Open your web browser and go to `http://localhost:8501` to use the app.

3. **Upload your Excel file**:
   
   Use the provided interface to upload an Excel file formatted as described below.

4. **Choose a GPT model**:
   
   Select the desired GPT model from the dropdown.

5. **Run the processing**:
   
   Click "Run" to start processing the file. Progress will be shown.

## Excel File Structure

### **Input Structure**:

The Excel file you upload should be structured as follows:

- **Row 1**: Contains headers (optional for your understanding, not used in processing).
- **Row 2**: Contains the prompts where `{parameter}` is a placeholder for dynamic content such as `{Fruit}`, `{Color}`, or `{Shape}`.
- **Column A (starting from Row 3)**: Contains the parameter names that identify each row (e.g., `Parameter 1`, `Parameter 2`).
- **Columns B and beyond (starting from Row 3)**: Contains the actual parameter values that will replace the `{parameter}` placeholders in the prompts (e.g., `Apple`, `Red`, `Round`).

#### Example:

|    A           |       B      |       C        |       D      |       E                     |       F                        |        G                       |
|----------------|--------------|----------------|--------------|-----------------------------|--------------------------------|--------------------------------|
| Index          | Fruit        | Color          | Shape        | Prompt 1                    | Prompt 2                       | Prompt 3                       |
| None           | None         | None           | None         | What is {Fruit}?            | Why is {Fruit} {Color}?        | Why is {Fruit} {Shape}?        |
| Parameter 1    | Apple        | Red            | Round        |                             |                                |                                |
| Parameter 2    | Banana       | Yellow         | Long         |                             |                                |                                |
| Parameter 3    | Grape        | Purple         | Ball         |                             |                                |                                |

### **Output Structure**:

After processing, the Excel file will be updated with the responses from GPT models in the cells corresponding to each parameter and prompt combination.

#### Example Output:

|    A           |       B      |       C        |       D      |       E                         |       F                               |        G                              |
|----------------|--------------|----------------|--------------|---------------------------------|---------------------------------------|---------------------------------------|
| Index          | Fruit        | Color          | Shape        | Prompt 1                        | Prompt 2                              | Prompt 3                              |
| None           | None         | None           | None         | What is {Fruit}?                | Why is {Fruit} {Color}?               | Why is {Fruit} {Shape}?               |
| Parameter 1    | Apple        | Red            | Round        | An edible fruit.                | Due to the presence of anthocyanins.  | Apple is round for uniform growth.    |
| Parameter 2    | Banana       | Yellow         | Long         | A tropical fruit.               | Due to the presence of carotenoids.   | Bananas grow long for better sunlight |
| Parameter 3    | Grape        | Purple         | Ball         | A small, juicy fruit.           | Due to anthocyanin pigments.          | Grapes are round for easy consumption.|

### **URL Fetching**:

If any parameter value starts with http:// or https://, TableGen AI will automatically fetch the content of the web page, extract the unformatted text, and return it as the parameter value.


### **Explanation**:

- **Row 2**: Defines the prompts with `{parameter}` as a placeholder. This allows for dynamic content to be inserted based on the parameters in the subsequent rows.
- **Column A**: Contains the parameter names (e.g., `Parameter 1`, `Parameter 2`) which serve as identifiers for each set of values.
- **Columns B, C, and D (starting from Row 3)**: These columns contain the actual values that will replace the placeholders `{Fruit}`, `{Color}`, and `{Shape}` in the prompts.
- **Columns E, F, and G**: These columns contain the processed responses generated by the selected GPT model. The `{parameter}` placeholders in the prompts are dynamically replaced by the actual values from Columns B, C, and D.

## When to use it?

- **Dynamic Prompting**: By structuring your file in this way, you can create highly dynamic and personalized content for each parameter.
- **Scalability**: This structure allows you to easily scale the content generation by adding more parameters (rows) or prompts (columns).
- **Efficiency**: The app automates the replacement of parameters in prompts and generates responses, saving time and ensuring consistency.



