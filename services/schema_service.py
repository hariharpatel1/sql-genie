import logging
from typing import Dict, List, Any, Optional, Set
import time
from core.database import RedshiftConnector

logger = logging.getLogger(__name__)

class SchemaService:
    """Service for retrieving and managing database schema information"""
    
    def __init__(self, db_connector: RedshiftConnector):
        self.db_connector = db_connector
        self.schema_cache = {}
        self.cache_timestamp = 0
        self.cache_ttl = 600  # Cache TTL in seconds (10 minutes)
    
    def get_all_tables(self, schema: str = "public") -> List[str]:
        """Get all tables in the specified schema"""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = %s
        AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
        
        try:
            df, _ = self.db_connector.execute_query(query, (schema,))
            return df['table_name'].tolist()
        except Exception as e:
            logger.error(f"Error retrieving tables: {e}")
            return []
    
    def get_table_schema(self, table_name: str, schema: str = "public") -> List[Dict[str, Any]]:
        """Get schema information for a specific table"""
        # First get the columns
        columns_query = """
        SELECT 
            column_name, 
            data_type,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            is_nullable
        FROM 
            information_schema.columns
        WHERE 
            table_schema = %s
            AND table_name = %s
        ORDER BY 
            ordinal_position;
        """
        
        # Then get comments/descriptions
        comments_query = """
        SELECT
            col.column_name,
            pg_description.description
        FROM
            information_schema.columns col
        LEFT JOIN
            pg_class ON pg_class.relname = col.table_name
        LEFT JOIN
            pg_attribute ON pg_attribute.attrelid = pg_class.oid
            AND pg_attribute.attname = col.column_name
        LEFT JOIN
            pg_description ON pg_description.objoid = pg_class.oid
            AND pg_description.objsubid = pg_attribute.attnum
        WHERE
            col.table_schema = %s
            AND col.table_name = %s;
        """
        
        try:
            columns_df, _ = self.db_connector.execute_query(columns_query, (schema, table_name))
            comments_df, _ = self.db_connector.execute_query(comments_query, (schema, table_name))
            
            # Convert DataFrames to dicts for easier lookup
            comments_dict = dict(zip(comments_df['column_name'], comments_df['description']))
            
            # Combine the information
            schema_info = []
            for _, row in columns_df.iterrows():
                column_info = row.to_dict()
                column_name = column_info['column_name']
                
                # Add description if available
                column_info['description'] = comments_dict.get(column_name, '')
                
                # Format the data type with precision if applicable
                if column_info['character_maximum_length'] is not None:
                    column_info['formatted_data_type'] = f"{column_info['data_type']}({column_info['character_maximum_length']})"
                elif column_info['numeric_precision'] is not None and column_info['numeric_scale'] is not None:
                    column_info['formatted_data_type'] = f"{column_info['data_type']}({column_info['numeric_precision']},{column_info['numeric_scale']})"
                else:
                    column_info['formatted_data_type'] = column_info['data_type']
                
                schema_info.append(column_info)
            
            return schema_info
        except Exception as e:
            logger.error(f"Error retrieving schema for table {table_name}: {e}")
            return []
    
    def get_table_primary_keys(self, table_name: str, schema: str = "public") -> List[str]:
        """Get primary key columns for a table"""
        query = """
        SELECT
            c.column_name
        FROM
            information_schema.table_constraints tc
        JOIN
            information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name)
        JOIN
            information_schema.columns AS c ON c.table_schema = tc.constraint_schema
            AND tc.table_name = c.table_name AND ccu.column_name = c.column_name
        WHERE
            tc.constraint_type = 'PRIMARY KEY'
            AND tc.table_schema = %s
            AND tc.table_name = %s;
        """
        
        try:
            df, _ = self.db_connector.execute_query(query, (schema, table_name))
            return df['column_name'].tolist()
        except Exception as e:
            logger.error(f"Error retrieving primary keys for table {table_name}: {e}")
            return []
    
    def get_table_foreign_keys(self, table_name: str, schema: str = "public") -> List[Dict[str, str]]:
        """Get foreign key relationships for a table"""
        query = """
        SELECT
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM
            information_schema.table_constraints AS tc
        JOIN
            information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN
            information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE
            tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = %s
            AND tc.table_name = %s;
        """
        
        try:
            df, _ = self.db_connector.execute_query(query, (schema, table_name))
            foreign_keys = []
            for _, row in df.iterrows():
                foreign_keys.append({
                    'column_name': row['column_name'],
                    'foreign_table': row['foreign_table_name'],
                    'foreign_column': row['foreign_column_name']
                })
            return foreign_keys
        except Exception as e:
            logger.error(f"Error retrieving foreign keys for table {table_name}: {e}")
            return []
    
    def get_all_schemas(self, schema: str = "public", tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get schemas for all tables or specified tables
        Uses caching to avoid frequent database calls
        """
        current_time = time.time()
        
        # Use cached schema if it's still valid and we're requesting all tables
        if (tables is None and 
            self.schema_cache and 
            current_time - self.cache_timestamp < self.cache_ttl):
            return self.schema_cache
        
        schema_info = {}
        
        # Get all tables if not specified
        if tables is None:
            tables = self.get_all_tables(schema)
        
        # Get schema for each table
        for table_name in tables:
            table_schema = self.get_table_schema(table_name, schema)
            primary_keys = self.get_table_primary_keys(table_name, schema)
            foreign_keys = self.get_table_foreign_keys(table_name, schema)
            
            schema_info[table_name] = {
                'columns': table_schema,
                'primary_keys': primary_keys,
                'foreign_keys': foreign_keys
            }
        
        # Update cache if we're getting all tables
        if tables is None:
            self.schema_cache = schema_info
            self.cache_timestamp = current_time
        
        return schema_info
    
    def get_table_relationships(self) -> Dict[str, Set[str]]:
        """
        Get relationships between tables based on foreign keys
        Returns a dict where keys are table names and values are sets of related tables
        """
        tables = self.get_all_tables()
        relationships = {table: set() for table in tables}
        
        for table in tables:
            foreign_keys = self.get_table_foreign_keys(table)
            for fk in foreign_keys:
                foreign_table = fk['foreign_table']
                # Add relationship in both directions
                relationships[table].add(foreign_table)
                if foreign_table in relationships:
                    relationships[foreign_table].add(table)
        
        return relationships