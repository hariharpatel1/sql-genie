import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import json
import time
import re
from datetime import datetime, date
from decimal import Decimal
from services.data_context_service import data_context_service
from services.llm_service import llm_service

logger = logging.getLogger(__name__)

# Directly include the JSON encoder in this file
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle pandas Timestamps and other special types"""
    def default(self, obj):
        # Handle pandas Timestamp objects
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        
        # Handle numpy integers and floats
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
            
        # Handle datetime objects
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
            
        # Handle Decimal objects
        if isinstance(obj, Decimal):
            return float(obj)
            
        # Let the base class handle other types or raise TypeError
        return super().default(obj)

def is_timestamp(obj):
    """Check if an object is a pandas Timestamp"""
    return isinstance(obj, pd.Timestamp)

def safe_json_dumps(data):
    """Safely convert data to JSON string, handling special types"""
    try:
        return json.dumps(data, cls=CustomJSONEncoder)
    except Exception as e:
        logger.error(f"Error in json serialization: {str(e)}")
        # For DataFrames, try converting to string representation
        if isinstance(data, pd.DataFrame):
            return str(data.to_dict())
        # For other types, convert to string
        return str(data)

class LLMDataQueryService:
    """Service for using LLM to query and analyze in-memory data without SQL"""
    
    def __init__(self):
        """Initialize the LLM data query service"""
        pass
    
    def can_use_in_memory_data(self, conversation_id: str, query: str) -> Tuple[bool, Optional[str]]:
        """
        Determine if a query can be answered using in-memory data
        
        Args:
            conversation_id: ID of the conversation
            query: User's natural language query or SQL query
            
        Returns:
            Tuple of (can_use_in_memory, reason)
        """
        # Get available tables in the conversation context
        available_tables = data_context_service.get_all_tables(conversation_id)
        if not available_tables:
            return False, "No in-memory tables available"
        
        try:
            # Create a context summary for the LLM to evaluate
            context_summary = data_context_service.get_context_summary(conversation_id)
            tables_info = []
            
            for table in context_summary.get('tables', []):
                # Convert any non-serializable objects in the sample
                safe_sample = []
                for row in table.get('sample', []):
                    safe_row = {}
                    for k, v in row.items():
                        if is_timestamp(v):
                            safe_row[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            safe_row[k] = v
                    safe_sample.append(safe_row)
                
                tables_info.append({
                    'name': table['name'],
                    'rows': table['rows'],
                    'columns': table['column_names'],
                    'sample': safe_sample
                })
            
            # If there are no tables with data, we can't use in-memory data
            if not tables_info:
                return False, "No in-memory tables with data available"
            
            # Extract table names from SQL query or natural language query
            if query.strip().upper().startswith("SELECT"):
                # This looks like an SQL query
                referenced_tables = self.extract_table_names(query)
                
                # Check if all referenced tables are available in memory
                missing_tables = [table for table in referenced_tables 
                                if table not in available_tables and 
                                table.lower() not in [t.lower() for t in available_tables.keys()]]
                
                if missing_tables:
                    return False, f"Tables not in memory: {', '.join(missing_tables)}"
                
                # Simple check: if we have all tables referenced, we can try to use in-memory data
                return True, "All referenced tables are available in memory"
            else:
                # This is a natural language query, use the LLM to evaluate
                # Use safe_json_dumps to serialize tables_info
                tables_info_str = safe_json_dumps(tables_info)
                
                prompt = f"""
                I have the following data available in memory:
                {tables_info_str}
                
                The user has asked the following question:
                "{query}"
                
                Based solely on the available in-memory data, can this question be answered? 
                Return a JSON with the following format:
                {{
                  "can_answer": true/false,
                  "reason": "explanation",
                  "relevant_tables": ["table1", "table2"]
                }}
                
                Only respond with the JSON.
                """
                
                response = llm_service.generate_text(prompt)
                
                try:
                    # Parse the JSON response
                    result = json.loads(response)
                    can_answer = result.get('can_answer', False)
                    reason = result.get('reason', 'Unknown reason')
                    
                    if can_answer:
                        logger.info(f"LLM determined query can be answered with in-memory data: {reason}")
                        return True, reason
                    else:
                        logger.info(f"LLM determined query cannot be answered with in-memory data: {reason}")
                        return False, reason
                        
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse LLM response as JSON: {response}")
                    return False, "Could not determine if in-memory data is sufficient"
                    
        except Exception as e:
            logger.error(f"Error determining if in-memory data can be used: {str(e)}")
            return False, f"Error analyzing query for in-memory execution: {str(e)}"
    
    def extract_table_names(self, sql_query: str) -> List[str]:
        """
        Extract table names from an SQL query (compatibility with query_optimizer_service)
        
        Args:
            sql_query: SQL query to analyze
            
        Returns:
            List of table names
        """
        table_names = []
        
        try:
            # Use regex for simple extraction
            # Match for "FROM table_name" pattern
            from_matches = re.finditer(r'FROM\s+([^\s,;()]+)', sql_query, re.IGNORECASE)
            for match in from_matches:
                table_names.append(match.group(1))
            
            # Match for "JOIN table_name" pattern
            join_matches = re.finditer(r'JOIN\s+([^\s,;()]+)', sql_query, re.IGNORECASE)
            for match in join_matches:
                table_names.append(match.group(1))
            
            # Clean up any schema prefixes
            table_names = [name.split('.')[-1] if '.' in name else name for name in table_names]
            
            return list(set(table_names))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting table names: {str(e)}")
            return []
    
    def enhance_join_query(self, sql_query: str, schema_info: Dict[str, Any]) -> str:
        """
        Enhance a join query with proper column mapping (compatibility with query_optimizer_service)
        
        Args:
            sql_query: SQL query with joins to enhance
            schema_info: Schema information with table structures
            
        Returns:
            Enhanced SQL query
        """
        # This is a simplified implementation to maintain compatibility
        # We'll use LLM to enhance the join query
        
        # Only enhance if it's a join query without ON condition
        if 'JOIN' not in sql_query.upper() or 'ON' in sql_query.upper():
            return sql_query
        
        try:
            # Extract the tables involved
            tables = self.extract_table_names(sql_query)
            if len(tables) < 2:
                return sql_query
            
            # Create schema information for the LLM
            schema_info_str = ""
            for table_name, table_schema in schema_info.items():
                schema_info_str += f"Table: {table_name}\n"
                if isinstance(table_schema, dict) and 'columns' in table_schema:
                    columns = table_schema['columns']
                    for col in columns:
                        if isinstance(col, dict) and 'column_name' in col:
                            schema_info_str += f"- {col['column_name']}: {col.get('data_type', 'unknown')}\n"
                schema_info_str += "\n"
            
            # Ask the LLM to enhance the join query
            prompt = f"""
            You are a SQL expert. Enhance the following SQL query by adding proper JOIN conditions.
            
            SQL Query:
            {sql_query}
            
            Schema Information:
            {schema_info_str}
            
            Add appropriate ON conditions to the JOINs based on the schema information.
            Only return the enhanced SQL query with no additional text or explanation.
            """
            
            enhanced_sql = llm_service.generate_text(prompt)
            
            # Clean up the response (remove markdown, etc.)
            enhanced_sql = enhanced_sql.replace('```sql', '').replace('```', '').strip()
            
            return enhanced_sql
        except Exception as e:
            logger.error(f"Error enhancing join query: {str(e)}")
            return sql_query  # Return original if enhancement fails
    
    def execute_in_memory_query(self, conversation_id: str, query: str) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Use the LLM to comprehensively query and analyze all available conversation data
        
        Args:
            conversation_id: ID of the conversation
            query: User's natural language query
            
        Returns:
            Tuple of (result_dataframe, response_text)
        """
        start_time = time.time()
        
        try:
            # Get full context summary
            context_summary = data_context_service.get_context_summary(conversation_id)
            
            # Prepare comprehensive data context for LLM
            comprehensive_context = {
                "tables": {},
                "conversation_history": [],
                "variables": context_summary.get('variables', {})
            }
            
            # Collect all available tables
            all_tables = data_context_service.get_all_tables(conversation_id)
            
            # Comprehensively process all tables
            for name, df in all_tables.items():
                # Serialize full table information
                comprehensive_context["tables"][name] = {
                    "schema": list(df.columns),
                    "row_count": len(df),
                    "full_data": self._serialize_dataframe(df)
                }
            
            # Include conversation history
            history_entries = context_summary.get('history', [])
            comprehensive_context["conversation_history"] = [
                entry.get('summary', '') for entry in history_entries[-10:]  # Last 10 entries
            ]
            
            # Create a comprehensive prompt that includes all available context
            prompt = f"""
            Comprehensive Conversation Context:
            {safe_json_dumps(comprehensive_context)}

            Current User Query: "{query}"

            Analysis and Response Requirements:
            1. Thoroughly analyze the entire conversation context
            2. Identify all relevant data sources and tables
            3. Perform comprehensive analysis across multiple tables if needed
            4. Provide a detailed, insightful response that:
            a) Directly answers the user's query
            b) Explains the data sources and reasoning
            c) Highlights any interesting insights or patterns
            d) Suggests any further actions or clarifications if necessary

            Response Format:
            - Provide a clear, narrative response
            - Include any calculated results or statistics
            - If possible, generate a summary table in markdown format
            - Explain your approach to answering the query
            """
            
            # Get the comprehensive response from the LLM
            response = llm_service.generate_text(prompt)
            
            # Try to extract any tabular data from the response
            result_df = self._extract_table_from_response(response)
            
            # Record the query in context history
            context = data_context_service.get_or_create_context(conversation_id)
            context['history'].append({
                'action': 'comprehensive_llm_data_query',
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'duration': time.time() - start_time,
                'summary': f"Performed comprehensive analysis across {len(all_tables)} tables"
            })
            
            return result_df, response
            
        except Exception as e:
            logger.error(f"Error in comprehensive data query: {str(e)}")
            return None, f"I encountered an error while analyzing the conversation data: {str(e)}"

    def _serialize_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Serialize DataFrame to a list of dictionaries with safe type conversion
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of serialized rows
        """
        serialized_data = []
        for _, row in df.iterrows():
            safe_row = {}
            for k, v in row.items():
                if is_timestamp(v):
                    safe_row[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(v, (np.integer, np.int64)):
                    safe_row[k] = int(v)
                elif isinstance(v, (np.floating, np.float64)):
                    safe_row[k] = float(v)
                else:
                    safe_row[k] = str(v)  # Convert other types to string to ensure serializability
            serialized_data.append(safe_row)
        return serialized_data

    def _extract_table_from_response(self, response: str) -> Optional[pd.DataFrame]:
        """
        Attempt to extract tabular data from an LLM response
        
        Args:
            response: LLM response text
            
        Returns:
            DataFrame if tabular data is found, otherwise None
        """
        try:
            # Look for markdown tables
            import re
            
            # Pattern for markdown tables
            table_pattern = r'\|([^\|]+)\|([^\|]+)\|.*\n\|[\s\-\:]+\|[\s\-\:]+\|[^\n]*\n((.*\n)*?)'
            table_match = re.search(table_pattern, response)
            
            if table_match:
                # Extract the table content
                table_text = table_match.group(0)
                lines = table_text.strip().split('\n')
                
                # Get headers
                headers = [h.strip() for h in lines[0].split('|')[1:-1]]
                
                # Skip the separator line (|---|---|)
                data_rows = []
                for line in lines[2:]:
                    if '|' in line:
                        cells = [cell.strip() for cell in line.split('|')[1:-1]]
                        if len(cells) == len(headers):
                            data_rows.append(cells)
                
                # Create DataFrame
                if data_rows:
                    return pd.DataFrame(data_rows, columns=headers)
            
            # Look for JSON data in the response
            json_pattern = r'```json\n(.*?)\n```'
            json_matches = re.findall(json_pattern, response, re.DOTALL)
            
            if json_matches:
                for json_text in json_matches:
                    try:
                        data = json.loads(json_text)
                        if isinstance(data, list) and data and isinstance(data[0], dict):
                            return pd.DataFrame(data)
                    except json.JSONDecodeError:
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting table from response: {str(e)}")
            return None
    
    def answer_with_context(self, conversation_id: str, query: str, llm_context: Dict[str, Any] = None) -> str:
        """
        Generate a comprehensive answer using in-memory data context
        
        Args:
            conversation_id: ID of the conversation
            query: User's natural language query
            llm_context: Additional context for the LLM
            
        Returns:
            Formatted response text
        """
        try:
            # Get context summary
            context_summary = data_context_service.get_context_summary(conversation_id)
            
            # Extract information about available data
            tables_info = []
            for table in context_summary.get('tables', []):
                table_info = {
                    'name': table['name'],
                    'rows': table['rows'],
                    'columns': table['column_names']
                }
                
                # Get a few sample rows for reference
                df = data_context_service.get_table(conversation_id, table['name'])
                if df is not None and not df.empty:
                    # Convert to a serializable format
                    sample_rows = []
                    for _, row in df.head(3).iterrows():
                        safe_row = {}
                        for k, v in row.items():
                            if is_timestamp(v):
                                safe_row[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                            elif isinstance(v, (np.integer, np.int64)):
                                safe_row[k] = int(v)
                            elif isinstance(v, (np.floating, np.float64)):
                                safe_row[k] = float(v)
                            else:
                                safe_row[k] = v
                        sample_rows.append(safe_row)
                    table_info['sample'] = sample_rows
                
                tables_info.append(table_info)
            
            # Get conversation history for context
            conversation_history = []
            for entry in context_summary.get('history', []):
                if entry['action'] in ['store_data', 'register_table', 'llm_data_query']:
                    conversation_history.append(entry['summary'])
            
            # Combine everything into a rich context for the LLM
            prompt = f"""
            You are an AI assistant analyzing data for the user. You have access to the following in-memory data tables:
            {safe_json_dumps(tables_info)}
            
            Recent context history:
            {safe_json_dumps(conversation_history[-5:] if conversation_history else [])}
            
            The user is asking: "{query}"
            
            Based on the available data, provide a comprehensive and insightful answer. 
            If the data doesn't contain information to fully answer the query, clarify what's missing.
            
            Format your response as a polished, informative answer that includes:
            1. A direct answer to the question
            2. Relevant statistics, trends or patterns in the data
            3. Additional insights that might be valuable to the user
            
            Make your response conversational and helpful.
            """
            
            # Add any additional context provided
            if llm_context:
                # Use safe_json_dumps for additional context
                safe_context = {}
                if "database_results" in llm_context and isinstance(llm_context["database_results"], list):
                    safe_results = []
                    for row in llm_context["database_results"]:
                        safe_row = {}
                        for k, v in row.items():
                            if is_timestamp(v):
                                safe_row[k] = str(v)
                            elif isinstance(v, (np.integer, np.int64)):
                                safe_row[k] = int(v)
                            elif isinstance(v, (np.floating, np.float64)):
                                safe_row[k] = float(v)
                            else:
                                safe_row[k] = v
                        safe_results.append(safe_row)
                    safe_context["database_results"] = safe_results
                else:
                    safe_context = llm_context
                
                prompt += f"\n\nAdditional context: {safe_json_dumps(safe_context)}"
            
            # Get the response from the LLM
            response = llm_service.generate_text(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating contextual answer: {str(e)}")
            return f"I encountered an error while analyzing the data: {str(e)}"


# Create a global instance
llm_data_query_service = LLMDataQueryService()