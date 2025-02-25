from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd


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
        schema_text = ""
        
        for table_name, schema in self.table_schemas.items():
            schema_text += f"Table: {table_name}\n"
            schema_text += "Columns:\n"
            
            for column in schema:
                name = column.get("column_name", "")
                data_type = column.get("data_type", "")
                nullable = column.get("is_nullable", "YES")
                description = column.get("description", "")
                
                schema_text += f"  - {name} ({data_type}"
                if nullable == "NO":
                    schema_text += ", NOT NULL"
                schema_text += f"): {description}\n"
            
            schema_text += "\n"
        
        return schema_text