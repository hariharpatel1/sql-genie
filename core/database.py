import os
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor
from typing import Optional, Dict, List, Any, Tuple
import logging
import time
from core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class RedshiftConnector:
    """
    A class for handling connections and queries to Amazon Redshift
    """
    
    def __init__(self, connection_params: Optional[Dict[str, str]] = None):
        """
        Initialize the RedshiftConnector with connection parameters
        
        Args:
            connection_params: Dictionary containing connection parameters.
                If None, will use settings from config.
        """
        self.connection_params = connection_params or {
            "dbname": settings.REDSHIFT_DBNAME,
            "user": settings.REDSHIFT_USER,
            "password": settings.REDSHIFT_PASSWORD,
            "host": settings.REDSHIFT_HOST,
            "port": settings.REDSHIFT_PORT,
            "connect_timeout": settings.REDSHIFT_CONNECT_TIMEOUT
        }
        
        # Validate connection parameters
        required_params = ["dbname", "user", "password", "host"]
        missing_params = [param for param in required_params if not self.connection_params.get(param)]
        
        if missing_params:
            logger.warning(f"Missing required connection parameters: {', '.join(missing_params)}")
    
    def connect(self) -> psycopg2.extensions.connection:
        """
        Establish a connection to Redshift
        
        Returns:
            psycopg2 connection object
        """
        try:
            conn = psycopg2.connect(**self.connection_params)
            return conn
        except Exception as e:
            logger.error(f"[RedshiftConnector] Error connecting to Redshift: {str(e)}")
            raise
    
    def execute_query(self, query: str, params: tuple = None, 
                      max_rows: int = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Execute a SQL query and return results as a DataFrame
        
        Args:
            query: SQL query string
            params: Parameters to substitute in the query
            max_rows: Maximum number of rows to return (None for all)
            
        Returns:
            Tuple of (DataFrame with results, list of column names)
        """
        conn = None
        start_time = time.time()
        
        try:
            conn = self.connect()
            
            # Set query timeout if specified
            if hasattr(settings, 'REDSHIFT_QUERY_TIMEOUT') and settings.REDSHIFT_QUERY_TIMEOUT:
                with conn.cursor() as cur:
                    cur.execute(f"SET statement_timeout TO {settings.REDSHIFT_QUERY_TIMEOUT * 1000};")
            
            # Use DictCursor to get column names
            cursor = conn.cursor(cursor_factory=DictCursor)
            
            # Limit the query if max_rows is specified and no LIMIT exists
            if max_rows is not None and "LIMIT" not in query.upper():
                query = f"{query} LIMIT {max_rows}"
            
            # Execute the query
            logger.info(f"Executing query: {query}")
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Get column names
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Fetch results
            results = cursor.fetchall()
            
            # Create DataFrame
            df = pd.DataFrame(results, columns=column_names)
            
            # Log execution time and row count
            execution_time = time.time() - start_time
            logger.info(f"Query executed in {execution_time:.2f}s, returned {len(df)} rows")
            
            return df, column_names
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[RedshiftConnector] Error executing query (after {execution_time:.2f}s): {str(e)}")
            # Return empty DataFrame with error info
            df = pd.DataFrame()
            raise
        finally:
            if conn:
                conn.close()
    
    def validate_sql(self, sql_query: str) -> Tuple[bool, str]:
        """
        Validate a SQL query without executing it
        
        Args:
            sql_query: SQL query string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # We'll use EXPLAIN to validate the query without executing it
        explain_query = f"EXPLAIN {sql_query}"
        
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute(explain_query)
            cursor.fetchall()  # Consume results
            conn.close()
            return True, ""
        except Exception as e:
            error_message = str(e)
            return False, error_message
    
    def get_query_cost_estimate(self, sql_query: str) -> Dict[str, Any]:
        """
        Get an estimate of the query cost
        
        Args:
            sql_query: SQL query string
            
        Returns:
            Dictionary with cost information
        """
        explain_query = f"EXPLAIN {sql_query}"
        
        try:
            df, _ = self.execute_query(explain_query)
            
            # Parse the EXPLAIN output to extract cost information
            cost_info = {
                "estimated_rows": 0,
                "estimated_cost": 0,
                "query_plan": df.iloc[:, 0].tolist() if not df.empty else []
            }
            
            # Try to extract row estimates from the query plan
            for plan_row in cost_info["query_plan"]:
                if "rows=" in plan_row:
                    try:
                        cost_info["estimated_rows"] = int(
                            plan_row.split("rows=")[1].split(" ")[0]
                        )
                        break
                    except (ValueError, IndexError):
                        pass
            
            return cost_info
        except Exception as e:
            logger.error(f"[RedshiftConnector] Error getting query cost estimate: {str(e)}")
            return {"error": str(e)}
    
    def enforce_query_limits(self, sql_query: str, max_rows: int = None) -> str:
        """
        Enforce limits on a query to prevent excessive resource usage
        
        Args:
            sql_query: Original SQL query
            max_rows: Maximum number of rows to return
            
        Returns:
            Modified SQL query with limits
        """
        if max_rows is None:
            max_rows = settings.REDSHIFT_MAX_ROWS
            
        # Simple implementation - just add a LIMIT clause if not present
        if "LIMIT" not in sql_query.upper():
            sql_query = f"{sql_query} LIMIT {max_rows}"
        
        return sql_query
    
    def test_connection(self) -> bool:
        """
        Test the connection to Redshift
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            conn = self.connect()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"[RedshiftConnector] Connection test failed: {str(e)}")
            return False