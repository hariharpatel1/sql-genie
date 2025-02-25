# SQLGenie
“Your wish for SQL queries, granted by AI.”

A conversational AI agent that helps query SQL databases using natural language. Built with Python, LangChain, LangGraph, Azure OpenAI, and Streamlit.

## Features

- Convert natural language queries to SQL
- Validate user queries and ask for missing information
- Execute SQL queries against a sql database
- Explain query results in natural language
- Suggest follow-up questions
- Remember conversation context
- Show raw SQL and query results

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sql-genie.git
   cd sql-genie
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your configuration:
   ```
   # Azure OpenAI settings
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2023-05-15
   AZURE_OPENAI_DEPLOYMENT=your_deployment_name

   # Redshift connection settings
   REDSHIFT_HOST=your_redshift_host
   REDSHIFT_PORT=5439
   REDSHIFT_DATABASE=your_database
   REDSHIFT_USER=your_username
   REDSHIFT_PASSWORD=your_password

   # Query settings
   MAX_QUERY_ROWS=1000
   MAX_QUERY_TIME=30

   # Application settings
   DEBUG=False
   ```

4. Create the necessary directory structure:
   ```bash
   mkdir -p models agents tools memory utils
   touch models/__init__.py agents/__init__.py tools/__init__.py memory/__init__.py utils/__init__.py
   ```

5. Copy the code files from this project into their respective directories.

## Running the Application

To start the application, run:

```bash
streamlit run main.py
```

The application will start a web server and open in your default browser.

## Usage

1. Enter your question in natural language in the text box.
2. Click "Ask" to submit your query.
3. The agent will:
   - Validate your query
   - Generate SQL
   - Execute the query
   - Explain the results
4. You can view the generated SQL and raw results by expanding the sections below the response.
5. Click on suggested follow-up questions to continue the conversation.

## Example Queries

- "Show me all transactions with amount greater than 1000 in the last month"
- "What are the top 5 payees by total transaction amount?"
- "How many failed transactions do we have per day in the last week?"
- "Show me ledger entries with status 'completed' for account number starting with '123'"

## Requirements

```
python>=3.8
streamlit
langchain
azure-openai
pandas
psycopg2-binary
python-dotenv
pydantic>=2.0.0
```

## License

MIT License

## Author


---

Feel free to customize this README as needed!
