import re
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import uuid
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class DataFrame:
    """A serializable wrapper around pandas DataFrames for storing in conversation context"""
    def __init__(self, df: pd.DataFrame, query: str, source: str = "redshift"):
        self.df = df
        self.created_at = datetime.utcnow()
        self.query = query
        self.source = source  # 'redshift', 'user_input', 'derived'
        self.id = str(uuid.uuid4())
        self.metadata = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary"""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "query": self.query,
            "source": self.source,
            "metadata": self.metadata,
            # Convert DataFrame to a serializable format
            "data": self.df.to_dict(orient='records')
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataFrame':
        """Create from a serialized dictionary"""
        df_data = data.get("data", [])
        df = pd.DataFrame(df_data)
        
        # Create a new DataFrame object
        df_obj = cls(df, data.get("query", ""), data.get("source", "redshift"))
        
        # Set the properties manually
        df_obj.id = data.get("id", str(uuid.uuid4()))
        df_obj.created_at = datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat()))
        df_obj.metadata = data.get("metadata", {})
        
        return df_obj


class DataContextService:
    """Service for managing and persisting data context within conversations"""
    
    def __init__(self):
        """Initialize the data context service"""
        # Main storage for conversation data contexts
        # Keys are conversation IDs, values are dictionaries containing:
        # - 'dataframes': List of DataFrame objects
        # - 'tables': Dict mapping table name/alias to DataFrames
        # - 'variables': Dict of variables set during conversation
        self.contexts = {}
    
    def get_or_create_context(self, conversation_id: str) -> Dict[str, Any]:
        """Get an existing context or create a new one for a conversation"""
        if conversation_id not in self.contexts:
            self.contexts[conversation_id] = {
                'dataframes': [],
                'tables': {},
                'variables': {},
                'history': []
            }
        return self.contexts[conversation_id]
    
    def store_query_result(self, conversation_id: str, df: pd.DataFrame, 
                          query: str, source: str = "redshift") -> str:
        """
        Store a new query result in the conversation context
        
        Args:
            conversation_id: ID of the conversation
            df: DataFrame with query results
            query: The SQL query or description of the data
            source: Source of the data ('redshift', 'user_input', 'derived')
            
        Returns:
            ID of the stored DataFrame
        """
        context = self.get_or_create_context(conversation_id)
        df_wrapper = DataFrame(df, query, source)
        
        # Add to dataframes list
        context['dataframes'].append(df_wrapper)
        
        # Add action to history
        context['history'].append({
            'action': 'store_data',
            'dataframe_id': df_wrapper.id,
            'timestamp': datetime.utcnow().isoformat(),
            'source': source,
            'summary': f"Stored {df.shape[0]} rows × {df.shape[1]} columns from {source}"
        })
        
        logger.info(f"Stored query result in conversation {conversation_id}, df_id: {df_wrapper.id}, shape: {df.shape}")
        return df_wrapper.id
    
    def register_table(self, conversation_id: str, table_name: str, 
                      df_id: str) -> bool:
        """
        Register a DataFrame as a named table in the conversation context
        
        Args:
            conversation_id: ID of the conversation
            table_name: Name to assign to the table
            df_id: ID of the DataFrame to register
            
        Returns:
            Success status
        """
        context = self.get_or_create_context(conversation_id)
        
        # Find the DataFrame by ID
        for df_wrapper in context['dataframes']:
            if df_wrapper.id == df_id:
                # Register the table
                context['tables'][table_name] = df_wrapper
                
                # Add action to history
                context['history'].append({
                    'action': 'register_table',
                    'table_name': table_name,
                    'dataframe_id': df_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'summary': f"Registered table '{table_name}' with {df_wrapper.df.shape[0]} rows"
                })
                
                logger.info(f"Registered table '{table_name}' in conversation {conversation_id}, df_id: {df_id}")
                return True
        
        logger.warning(f"Failed to register table '{table_name}', DataFrame {df_id} not found")
        return False
    
    def store_user_data(self, conversation_id: str, data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame], 
                      description: str) -> str:
        """
        Store user-provided data in the conversation context
        
        Args:
            conversation_id: ID of the conversation
            data: User data as a dict, list of dicts, or DataFrame
            description: Description of the data
            
        Returns:
            ID of the stored DataFrame
        """
        # Convert data to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, dict):
                # Single row as dict
                df = pd.DataFrame([data])
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                # List of dicts
                df = pd.DataFrame(data)
            else:
                raise ValueError("Data must be a DataFrame, dict, or list of dicts")
        else:
            df = data
        
        # Store the DataFrame
        df_id = self.store_query_result(conversation_id, df, description, source="user_input")
        
        logger.info(f"Stored user data in conversation {conversation_id}, df_id: {df_id}, shape: {df.shape}")
        return df_id
    
    def get_dataframe(self, conversation_id: str, df_id: str) -> Optional[pd.DataFrame]:
        """
        Get a DataFrame by ID from the conversation context
        
        Args:
            conversation_id: ID of the conversation
            df_id: ID of the DataFrame to retrieve
            
        Returns:
            The pandas DataFrame or None if not found
        """
        context = self.get_or_create_context(conversation_id)
        
        for df_wrapper in context['dataframes']:
            if df_wrapper.id == df_id:
                return df_wrapper.df
        
        return None
    
    def get_table(self, conversation_id: str, table_name: str) -> Optional[pd.DataFrame]:
        """
        Get a registered table by name from the conversation context
        
        Args:
            conversation_id: ID of the conversation
            table_name: Name of the table to retrieve
            
        Returns:
            The pandas DataFrame or None if not found
        """
        context = self.get_or_create_context(conversation_id)
        
        if table_name in context['tables']:
            return context['tables'][table_name].df
        
        # Check for a case-insensitive match
        for name, df_wrapper in context['tables'].items():
            if name.lower() == table_name.lower():
                return df_wrapper.df
        
        return None
    
    def get_all_tables(self, conversation_id: str) -> Dict[str, pd.DataFrame]:
        """
        Get all registered tables for a conversation
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Dictionary mapping table names to DataFrames
        """
        context = self.get_or_create_context(conversation_id)
        return {name: df_wrapper.df for name, df_wrapper in context['tables'].items()}
    
    def get_latest_dataframe(self, conversation_id: str) -> Optional[pd.DataFrame]:
        """
        Get the most recently added DataFrame in the conversation
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            The most recent DataFrame or None if no DataFrames exist
        """
        context = self.get_or_create_context(conversation_id)
        
        if not context['dataframes']:
            return None
        
        # Sort by created_at and return the latest
        latest = sorted(context['dataframes'], key=lambda df: df.created_at, reverse=True)[0]
        return latest.df
    
    def set_variable(self, conversation_id: str, name: str, value: Any) -> None:
        """
        Set a variable in the conversation context
        
        Args:
            conversation_id: ID of the conversation
            name: Variable name
            value: Variable value
        """
        context = self.get_or_create_context(conversation_id)
        context['variables'][name] = value
        
        # Add action to history
        context['history'].append({
            'action': 'set_variable',
            'variable_name': name,
            'timestamp': datetime.utcnow().isoformat(),
            'summary': f"Set variable '{name}'"
        })
        
        logger.info(f"Set variable '{name}' in conversation {conversation_id}")
    
    def get_variable(self, conversation_id: str, name: str, default: Any = None) -> Any:
        """
        Get a variable from the conversation context
        
        Args:
            conversation_id: ID of the conversation
            name: Variable name
            default: Default value if variable doesn't exist
            
        Returns:
            Variable value or default if not found
        """
        context = self.get_or_create_context(conversation_id)
        return context['variables'].get(name, default)
    
    def get_context_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get a summary of the conversation context
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Dictionary with context summary
        """
        context = self.get_or_create_context(conversation_id)
        logger.info(f"[DB] Retrieved context summary for conversation: {context}, conversation_id: {conversation_id}")

        # Create a summary of available tables
        tables_summary = []
        for name, df_wrapper in context['tables'].items():
            df = df_wrapper.df
            tables_summary.append({
                'name': name,
                'rows': df.shape[0],
                'columns': df.shape[1],
                'column_names': list(df.columns),
                'source': df_wrapper.source,
                'sample': df.head(2).to_dict(orient='records') if not df.empty else []
            })
        
        # Create a summary of available DataFrames
        dataframes_summary = []
        for df_wrapper in context['dataframes']:
            df = df_wrapper.df
            dataframes_summary.append({
                'id': df_wrapper.id,
                'rows': df.shape[0],
                'columns': df.shape[1],
                'column_names': list(df.columns),
                'source': df_wrapper.source,
                'created_at': df_wrapper.created_at.isoformat(),
                'query': df_wrapper.query,
                'sample': df.head(2).to_dict(orient='records') if not df.empty else []
            })
        
        return {
            'conversation_id': conversation_id,
            'tables': tables_summary,
            'dataframes': dataframes_summary,
            'variables': context['variables'],
            'history': context['history'][-10:] if context['history'] else []  # Last 10 history entries
        }
    
    def clear_context(self, conversation_id: str) -> None:
        """
        Clear all context data for a conversation
        
        Args:
            conversation_id: ID of the conversation
        """
        if conversation_id in self.contexts:
            del self.contexts[conversation_id]
            logger.info(f"Cleared context for conversation {conversation_id}")
    
    def serialize_context(self, conversation_id: str) -> Dict[str, Any]:
        """
        Serialize the context data for storage
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Serializable dictionary representation
        """
        if conversation_id not in self.contexts:
            return {}
        
        context = self.contexts[conversation_id]
        
        # Convert DataFrames to serializable format
        serialized_dfs = [df.to_dict() for df in context['dataframes']]
        
        # Convert tables dictionary
        serialized_tables = {
            name: df_wrapper.id for name, df_wrapper in context['tables'].items()
        }
        
        return {
            'dataframes': serialized_dfs,
            'tables': serialized_tables,
            'variables': context['variables'],
            'history': context['history']
        }
    
    def deserialize_context(self, conversation_id: str, data: Dict[str, Any]) -> None:
        """
        Restore context data from serialized format
        
        Args:
            conversation_id: ID of the conversation
            data: Serialized context data
        """
        if not data:
            return
        
        # Create new context
        context = {
            'dataframes': [],
            'tables': {},
            'variables': data.get('variables', {}),
            'history': data.get('history', [])
        }
        
        # Restore DataFrames
        df_map = {}  # Map of DataFrame IDs to DataFrame objects
        for df_data in data.get('dataframes', []):
            try:
                df_wrapper = DataFrame.from_dict(df_data)
                context['dataframes'].append(df_wrapper)
                df_map[df_wrapper.id] = df_wrapper
            except Exception as e:
                logger.error(f"Error deserializing DataFrame: {str(e)}")
        
        # Restore tables
        for name, df_id in data.get('tables', {}).items():
            if df_id in df_map:
                context['tables'][name] = df_map[df_id]
            else:
                logger.warning(f"Could not restore table '{name}', DataFrame {df_id} not found")
        
        # Save the restored context
        self.contexts[conversation_id] = context
        logger.info(f"Restored context for conversation {conversation_id}")

    def _extract_main_table_name(self, sql_query: str) -> Optional[str]:
        """Extract the main table name from an SQL query for context registration"""
        try:
            # Use regex for simple extraction
            # Match for "FROM table_name" pattern
            from_matches = re.finditer(r'FROM\s+([^\s,;()]+)', sql_query, re.IGNORECASE)
            table_names = [match.group(1) for match in from_matches]
            
            # Match for "JOIN table_name" pattern
            join_matches = re.finditer(r'JOIN\s+([^\s,;()]+)', sql_query, re.IGNORECASE)
            for match in join_matches:
                table_names.append(match.group(1))
            
            if not table_names:
                return "query_results"
            
            # Return the actual table name without modifying it
            # We want to maintain the original table name, including any schema prefix
            main_table = table_names[0]
            
            # If there are multiple tables (a join), you might want to indicate that
            if len(table_names) > 1:
                return f"{main_table}_joined"
            
            # Otherwise, just return the exact table name without adding "_results"
            return main_table
                
        except Exception as e:
            logger.error(f"Error extracting table name: {str(e)}")
            return "query_results"

# Create a global instance
data_context_service = DataContextService()