import logging
from typing import Dict, Any, Optional, List
from core.config import settings
import json
import time

# LangChain imports
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with LLMs"""
    
    def __init__(self):
        self._llm = None
    
    @property
    def llm(self):
        """Lazy-loaded LLM instance"""
        if self._llm is None:
            self._llm = self._initialize_llm()
        return self._llm
    
    def _initialize_llm(self):
        """Initialize the Azure OpenAI LLM based on settings"""
        return AzureChatOpenAI(
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS
        )
    
    def understand_query(self, query: str, table_schemas: str) -> Dict[str, Any]:
        """
        Analyze a user query to understand its intent
        
        Args:
            query: The user's natural language query
            table_schemas: String representation of available table schemas
            
        Returns:
            Dict with query understanding
        """
        start_time = time.time()
        
        prompt = ChatPromptTemplate.from_template(
            """You are a financial data analyst assistant that helps users query transaction data.
            
            Given the user's query, identify the following information:
            1. What tables are they interested in? (from the available tables in the schema)
            2. What columns are they looking for?
            3. What filters or conditions should be applied?
            4. Is there a specific time range mentioned?
            5. Is the user asking for aggregations or statistics?
            6. Is there anything missing from the query that I need to ask for clarification?
            
            Here are the available tables and their schemas:
            {table_schemas}
            
            User query: {query}
            
            Provide your analysis as JSON with the following structure:
            {{
                "tables": ["table1", "table2"],
                "columns": ["column1", "column2"],
                "filters": {{"column": "value"}},
                "time_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}},
                "aggregations": ["sum", "avg", "count"],
                "missing_info": ["specific piece of missing info"],
                "needs_clarification": true/false
            }}
            
            Only include fields that are relevant to the query.
            """
        )
        
        chain = (
            {"query": lambda x: x, "table_schemas": lambda _: table_schemas}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            result = chain.invoke(query)
            query_understanding = json.loads(result)
            
            execution_time = time.time() - start_time
            logger.debug(f"Query understanding completed in {execution_time:.2f}s")
            
            return query_understanding
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[LLMService] Error understanding query (after {execution_time:.2f}s): {str(e)}")
            # Return a basic understanding
            return {
                "error": str(e),
                "needs_clarification": True,
                "missing_info": ["There was an error processing your query"]
            }
    
    def generate_clarification_question(self, query: str, missing_info: List[str]) -> str:
        """
        Generate a clarification question for the user
        
        Args:
            query: The original user query
            missing_info: List of missing information elements
            
        Returns:
            A clarification question
        """
        prompt = ChatPromptTemplate.from_template(
            """Based on the user's query: "{query}"
            
            I've identified that we need more information to properly process this request.
            Specifically, we're missing: {missing_info}
            
            Craft a polite question asking for this specific information. Be brief and clear.
            """
        )
        
        chain = (
            {
                "query": lambda x: x, 
                "missing_info": lambda y: ", ".join(y)
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            result = chain.invoke({"query": query, "missing_info": missing_info})
            return result
        except Exception as e:
            logger.error(f"Error generating clarification: {str(e)}")
            return "I need some more information to answer your question. Could you provide more details?"
    
    def generate_sql(self, query_understanding: Dict[str, Any], table_schemas: str) -> str:
        """
        Generate SQL for Redshift based on query understanding
        
        Args:
            query_understanding: Dictionary with query understanding
            table_schemas: String representation of available table schemas
            
        Returns:
            SQL query string
        """
        prompt = ChatPromptTemplate.from_template(
            """You are a SQL expert assistant that helps generate SQL for Redshift.
            
            Based on the following query understanding, generate a SQL query for Redshift:
            {query_understanding}
            
            Here are the available tables and their schemas:
            {table_schemas}
            
            Important guidelines:
            1. Use proper Redshift SQL syntax
            2. Join tables correctly if multiple tables are involved
            3. Never query more than 100,000 rows to avoid overloading the database
            4. Use appropriate LIMIT clauses
            5. If a time range is provided, always use it to limit the results
            6. If no time range is provided, limit to the last 30 days by default
            7. Ensure all column names and table names are correct
            
            Return ONLY the SQL query without any explanation or markdown.
            """
        )
        
        chain = (
            {
                "query_understanding": lambda x: json.dumps(x, indent=2), 
                "table_schemas": lambda y: y
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            result = chain.invoke({"query_understanding": query_understanding, "table_schemas": table_schemas})
            return result.strip()
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            raise
    
    def formulate_response(self, query: str, sql_query: str, sql_results: Any) -> str:
        """
        Formulate a natural language response based on query results
        
        Args:
            query: Original user query
            sql_query: SQL query that was executed
            sql_results: Results from the query (DataFrame or error message)
            
        Returns:
            Natural language response
        """
        prompt = ChatPromptTemplate.from_template(
            """You are a financial data analyst assistant that helps users understand their transaction data.
            
            The user asked: "{query}"
            
            Based on the query, I ran the following SQL:
            ```sql
            {sql_query}
            ```
            
            And got the following results:
            ```
            {sql_results}
            ```
            
            Provide a clear, concise explanation of these results that directly addresses the user's query.
            Focus on insights and patterns in the data. If there are any notable trends or outliers, mention them.
            
            Format your response in a conversational tone, but include specific numbers and statistics from the results.
            """
        )
        
        chain = (
            {
                "query": lambda x: x, 
                "sql_query": lambda y: y,
                "sql_results": lambda z: str(z)
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            result = chain.invoke({
                "query": query, 
                "sql_query": sql_query,
                "sql_results": sql_results
            })
            return result
        except Exception as e:
            logger.error(f"Error formulating response: {str(e)}")
            return "I found some results, but encountered an error while analyzing them. Please try simplifying your query."


# Create a global instance
llm_service = LLMService()