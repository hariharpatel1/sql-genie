from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    """Model for the agent state during a conversation turn"""
    conversation_id: str
    current_query: str = ""
    query_understanding: Dict[str, Any] = {}
    sql_query: str = ""
    sql_results: Any = None
    needs_clarification: bool = False
    clarification_question: str = ""
    final_response: str = ""
    error: str = ""
    timing: Dict[str, float] = {}
    table_schemas: Dict[str, Any] = {}  # Dynamically loaded from Redshift

    use_in_memory_data: bool = False
    in_memory_results: Any = None
    can_answer_from_memory: bool = False
    
    class Config:
        # Allow arbitrary types (like pandas DataFrames)
        arbitrary_types_allowed = True
        
    def reset_for_new_query(self, query: str):
        """Reset the state for a new query"""
        self.current_query = query
        self.query_understanding = {}
        self.sql_query = ""
        self.sql_results = None
        self.needs_clarification = False
        self.clarification_question = ""
        self.final_response = ""
        self.error = ""
        self.timing = {}
        # Don't reset table_schemas as they remain valid
    
    def update_from_clarification(self, clarification_input: str):
        """Update the state after receiving clarification"""
        self.current_query = f"{self.current_query} {clarification_input}"
        self.needs_clarification = False
        self.clarification_question = ""
    
    def set_error(self, error_message: str):
        """Set an error message"""
        self.error = error_message
    
    def record_timing(self, step_name: str, execution_time: float):
        """Record timing for a step"""
        self.timing[step_name] = execution_time
    
    def get_formatted_schemas(self) -> str:
        """Get the table schemas in a format suitable for LLM prompts"""
        # Handle case where table_schemas is already a string
        if isinstance(self.table_schemas, str):
            return self.table_schemas
        
        # Handle case where table_schemas is empty
        if not self.table_schemas:
            return "No database schemas available."
        
        schema_text = ""
        
        for table_name, schema in self.table_schemas.items():
            logger.info(f"Formatting schema for table: {table_name}")
            
            # Handle different possible schema structures
            if isinstance(schema, str):
                # If schema is a string, just add it directly
                schema_text += f"Table: {table_name}\n{schema}\n\n"
                continue
            
            # Extract columns based on different possible dict structures
            columns = schema.get('columns', []) if isinstance(schema, dict) else schema
            
            schema_text += f"Table: events.{table_name}\n"
            schema_text += "Columns:\n"
            
            for column in columns:
                # Ensure column is a dictionary
                if not isinstance(column, dict):
                    continue
                
                try:
                    name = column.get("column_name", "UNKNOWN")
                    
                    # Handle data type conversion for complex types
                    if isinstance(column.get("data_type"), pd.Timestamp):
                        data_type = str(column.get("data_type"))
                    else:
                        data_type = column.get("data_type", "UNDEFINED")
                    
                    nullable = column.get("is_nullable", "YES")
                    description = column.get("description", "No description")
                    
                    # Convert Timestamp to string if needed
                    if isinstance(description, pd.Timestamp):
                        description = str(description)
                    
                    schema_text += f"  - {name} ({data_type}"
                    if nullable == "NO":
                        schema_text += ", NOT NULL"
                    schema_text += f"): {description}\n"
                except Exception as e:
                    logger.error(f"Error formatting column {column}: {e}")
                    continue
            
            schema_text += "\n"
        
        logger.info(f"Formatted table schemas fetched from Redshift")
        return schema_text or "No valid schema information found."