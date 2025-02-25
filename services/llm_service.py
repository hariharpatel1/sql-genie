import logging
from typing import Dict, Any, Optional, List, Union

import pandas as pd
from core.config import settings
import json
import time
import re

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableMap, RunnablePassthrough

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
    
    def generate_clarification_question(self, query: str, missing_info: List[str]) -> str:
        """
        Generate a clarification question for the user
        
        Args:
            query: The original user query
            missing_info: List of missing information elements
            
        Returns:
            A clarification question
        """
        prompt = PromptTemplate.from_template(
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
        prompt = PromptTemplate.from_template(
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
            
            prompt = PromptTemplate.from_template(
                """You are a UPI analyst assistant that helps users query UPI-related transaction data and merchant-specific data.
                
                Given the user's query, identify the following information:
                1. What tables are they interested in? (from the available tables in the schema)
                2. What columns are they looking for?
                3. What filters or conditions should be applied?
                4. Is there a specific time range mentioned?
                5. Is the user asking for aggregations or statistics?
                6. Is there anything missing from the query that I need to ask for clarification?
                7. ; is always at the last of the query and not in the middle of the query.
                
                Here are the available tables and their schemas:
                {table_schemas}
                
                User query: {query}
                
                Provide your analysis as a VALID JSON with the following structure:
                {{
                    "tables": ["table1", "table2"],
                    "columns": ["column1", "column2"],
                    "filters": {{"column": "value"}},
                    "time_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}},
                    "aggregations": ["sum", "avg", "count"],
                    "missing_info": ["specific piece of missing info"],
                    "needs_clarification": true/false
                }}
                
                IMPORTANT: Ensure the JSON is perfectly formatted and valid. 
                Only include fields that are actually relevant to the query.
                If unsure about any field, use null or an empty list/object.
                """
            )
            
            # Use RunnableMap to create the input dictionary
            runnable = RunnableMap({
                "query": RunnablePassthrough(),
                "table_schemas": lambda x: table_schemas
            }) | prompt | self.llm | StrOutputParser()
            
            try:
                result = runnable.invoke(query)
                logger.info(f"LLM response: {result}")
                
                # Attempt to parse JSON with explicit error handling
                try:
                    query_understanding = json.loads(result)
                except json.JSONDecodeError:
                    logger.error("[ServiceLLM] Could not parse JSON from the LLM response")
                    # If JSON parsing fails, try to clean the response
                    # Remove any text before or after the JSON
                    json_match = re.search(r'\{.*\}', result, re.DOTALL)
                    if json_match:
                        try:
                            query_understanding = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            logger.error("[ServiceLLM] Could not parse JSON from the cleaned LLM response")
                            raise ValueError("Could not parse JSON from the LLM response")
                    else:
                        raise ValueError("No valid JSON found in the LLM response")
                
                execution_time = time.time() - start_time
                logger.debug(f"Query understanding completed in {execution_time:.2f}s")
                
                return query_understanding
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[LLMService] Error understanding query (after {execution_time:.2f}s): {str(e)}")
                # Return a basic understanding with clarification needed
                return {
                    "tables": [],
                    "columns": [],
                    "filters": {},
                    "time_range": None,
                    "aggregations": [],
                    "missing_info": ["There was an error processing your query. Could you please rephrase or provide more details?"],
                    "needs_clarification": True,
                    "error": str(e)
            }
    
    def formulate_response(self, query: str, sql_query: str, sql_results: Union[pd.DataFrame, dict]) -> str:
        """
        Formulate a natural language response based on SQL query and results
        
        Args:
            query: The original user query
            sql_query: The SQL query that was executed
            sql_results: The results of the SQL query as a pandas DataFrame or dictionary
            
        Returns:
            A natural language response
            containing the user query, SQL query, SQL results, and a response to the user query
            also contain some insights about the result data in 4-5 lines
        """
        logger.info(f"[LLMService] Formulating response for query: {query}")
        prompt = PromptTemplate.from_template(
            """Given the following user query: "{query}"
            
            And the SQL query used to retrieve data: ```{sql_query}```
            
            And the results of the query:
            {sql_results}
            
            Provide a clear, response to the user's query based on the SQL results.
            Try to answer the user's question directly, and format the response nicely.
            don't forget to include some insights about the result data in 4-5 lines.
            quatify the result data in 4-5 lines.
            """
        )
        
        sql_results_str = sql_results.to_string() if isinstance(sql_results, pd.DataFrame) else str(sql_results)
        
        chain = (
            {
                "query": lambda x: x,
                "sql_query": lambda x: x,
                "sql_results": lambda x: sql_results_str
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            result = chain.invoke({"query": query, "sql_query": sql_query, "sql_results": sql_results})
            logger.info(f"[LLMService] Formulated response: {result}")
            return result
        except Exception as e:
            logger.error(f"Error formulating response: {str(e)}")
            raise

# Create a global instance
llm_service = LLMService()