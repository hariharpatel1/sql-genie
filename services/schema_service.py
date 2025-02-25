import logging
from typing import Dict, List, Any, Optional, Set, Tuple
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
        
        # Add caches for individual queries to avoid repeated calls
        self._tables_cache = None
        self._schema_name = None
        self._columns_cache = {}
        self._primary_keys_cache = {}
        self._foreign_keys_cache = {}
        self._empty_tables = set()  # Track tables with no columns
        
        # Flag to avoid repeated failures
        self._db_connection_failed = False
        
        # Maximum number of tables to process
        self.max_tables_to_process = 20
    
    def get_all_tables(self, schema: str ="events") -> Tuple[List[str], str]:
        """
        Get all tables in the specified schema
        
        Returns:
            Tuple of (table_list, schema_name)
        """
        # Return cached tables if available
        if self._tables_cache is not None and self._schema_name is not None:
            logger.info(f"Using cached list of {len(self._tables_cache)} tables from schema {self._schema_name}")
            return self._tables_cache, self._schema_name
            
        # If we already know the connection failed, don't try again
        if self._db_connection_failed:
            logger.debug("Not attempting to retrieve tables due to previous connection failure")
            return [], schema
            
        query = """
        SELECT table_schema, table_name 
        FROM information_schema.tables 
        WHERE table_type = 'BASE TABLE' 
        AND table_name NOT LIKE 'pg_%' 
        AND table_name NOT LIKE 'sql_%' 
        ORDER BY table_schema, table_name;
        """
        
        try:
            # Test connection first to avoid unnecessary queries
            if not self.db_connector.test_connection():
                self._db_connection_failed = True
                logger.warning("Database connection test failed")
                return [], schema
                
            df, _ = self.db_connector.execute_query(query)
            
            if df.empty:
                logger.warning("No tables found in database")
                return [], schema
                
            tables = df['table_name'].tolist()
            schemas = df['table_schema'].tolist()
            
            # Get the first schema if available
            found_schema = schemas[0] if schemas else schema
            
            logger.info(f"Retrieved {len(tables)} tables from schema {found_schema}: {', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}")
            
            # Cache the result
            self._tables_cache = tables
            self._schema_name = found_schema
            
            return tables, found_schema
            
        except Exception as e:
            logger.error(f"Error retrieving tables: {e}")
            self._db_connection_failed = True
            return [], schema
    
    def get_table_schema(self, table_name: str, schema: str ="events") -> List[Dict[str, Any]]:
        """Get schema information for a specific table"""
        # Return from cache if available
        cache_key = f"{schema}.{table_name}"
        if cache_key in self._columns_cache:
            logger.debug(f"Using cached schema for table {table_name}")
            return self._columns_cache[cache_key]
        
        # If we know this table has no columns, return empty list
        if table_name in self._empty_tables:
            logger.debug(f"Skipping known empty table {table_name}")
            return []
            
        # If we already know the connection failed, don't try again
        if self._db_connection_failed:
            logger.debug(f"Not attempting to retrieve schema for table {table_name} due to previous connection failure")
            return []
            
        logger.info(f"Retrieving schema for table {table_name} in schema {schema}")
        
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
        
        try:
            columns_df, _ = self.db_connector.execute_query(columns_query, (schema, table_name))
            
            # If no columns found, mark as empty and return empty list
            if columns_df.empty:
                logger.info(f"No columns found for table {table_name}, marking as empty")
                self._empty_tables.add(table_name)
                self._columns_cache[cache_key] = []
                return []
            
            # Generate descriptions based on column names if no comments available
            schema_info = []
            for _, row in columns_df.iterrows():
                column_info = row.to_dict()
                column_name = column_info['column_name']
                
                # Generate a description based on the column name
                if column_name.lower().endswith('_id'):
                    description = f"Identifier for {column_name.lower().replace('_id', '')}"
                elif 'name' in column_name.lower():
                    description = f"Name field for {column_name.lower().replace('_name', '')}"
                elif 'date' in column_name.lower() or 'time' in column_name.lower():
                    description = f"Timestamp for {column_name.lower()}"
                else:
                    description = f"{column_name.lower().replace('_', ' ')} information"
                    
                column_info['description'] = description
                
                # Format the data type with precision if applicable
                if column_info['character_maximum_length'] is not None:
                    column_info['formatted_data_type'] = f"{column_info['data_type']}({column_info['character_maximum_length']})"
                elif column_info['numeric_precision'] is not None and column_info['numeric_scale'] is not None:
                    column_info['formatted_data_type'] = f"{column_info['data_type']}({column_info['numeric_precision']},{column_info['numeric_scale']})"
                else:
                    column_info['formatted_data_type'] = column_info['data_type']
                
                schema_info.append(column_info)
            
            logger.info(f"Retrieved {len(schema_info)} columns for table {table_name}")
            
            # Cache and return the result
            self._columns_cache[cache_key] = schema_info
            return schema_info
        except Exception as e:
            logger.error(f"Error retrieving schema for table {table_name}: {e}")
            self._columns_cache[cache_key] = []
            return []
    
    def get_table_primary_keys(self, table_name: str, schema: str ="events") -> List[str]:
        """Get primary key columns for a table"""
        # Skip empty tables
        if table_name in self._empty_tables:
            return []
            
        # Return from cache if available
        cache_key = f"{schema}.{table_name}"
        if cache_key in self._primary_keys_cache:
            return self._primary_keys_cache[cache_key]
            
        logger.info(f"Retrieving primary keys for table {table_name} in schema {schema}")
        
        # If we already know the connection failed, use a heuristic approach
        if self._db_connection_failed:
            logger.debug(f"Not querying for primary keys for table {table_name} due to previous connection failure")
            
            # Try to infer primary keys from column names (heuristic)
            table_schema = self.get_table_schema(table_name, schema)
            inferred_pks = []
            
            for column in table_schema:
                col_name = column.get('column_name', '').lower()
                if col_name == 'id' or col_name == f"{table_name.lower()}_id":
                    inferred_pks.append(column.get('column_name'))
                    
            self._primary_keys_cache[cache_key] = inferred_pks
            logger.info(f"Inferred {len(inferred_pks)} primary keys for table {table_name}")
            return inferred_pks
        
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
            primary_keys = df['column_name'].tolist() if not df.empty else []
            
            # If no primary keys found through constraints, try to infer from column names
            if not primary_keys:
                table_schema = self.get_table_schema(table_name, schema)
                for column in table_schema:
                    col_name = column.get('column_name', '').lower()
                    if col_name == 'id' or col_name == f"{table_name.lower()}_id":
                        primary_keys.append(column.get('column_name'))
            
            logger.info(f"Found {len(primary_keys)} primary keys for table {table_name}")
            
            # Cache and return the result
            self._primary_keys_cache[cache_key] = primary_keys
            return primary_keys
        except Exception as e:
            logger.error(f"Error retrieving primary keys for table {table_name}: {e}")
            self._primary_keys_cache[cache_key] = []
            return []
    
    def get_table_foreign_keys(self, table_name: str, schema: str ="events") -> List[Dict[str, str]]:
        """Get foreign key relationships for a table"""
        # Skip empty tables
        if table_name in self._empty_tables:
            return []
            
        # Return from cache if available
        cache_key = f"{schema}.{table_name}"
        if cache_key in self._foreign_keys_cache:
            return self._foreign_keys_cache[cache_key]
        
        logger.info(f"Retrieving foreign keys for table {table_name} in schema {schema}")
        
        # If we already know the connection failed, use a heuristic approach
        if self._db_connection_failed:
            logger.debug(f"Not querying for foreign keys for table {table_name} due to previous connection failure")
            
            # Try to infer foreign keys from column names (heuristic)
            table_schema = self.get_table_schema(table_name, schema)
            inferred_fks = []
            
            for column in table_schema:
                col_name = column.get('column_name', '').lower()
                if col_name.endswith('_id') and col_name != 'id' and col_name != f"{table_name.lower()}_id":
                    # Extract the potential referenced table
                    ref_table = col_name[:-3]  # Remove '_id'
                    
                    # Check if this table exists in our schema
                    all_tables, _ = self.get_all_tables(schema)
                    all_tables_lower = [t.lower() for t in all_tables]
                    if ref_table in all_tables_lower:
                        inferred_fks.append({
                            'column_name': column.get('column_name'),
                            'foreign_table': ref_table,
                            'foreign_column': f"{ref_table}_id"
                        })
                        
            self._foreign_keys_cache[cache_key] = inferred_fks
            logger.info(f"Inferred {len(inferred_fks)} foreign keys for table {table_name}")
            return inferred_fks
        
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
            
            if df.empty:
                # Try to infer foreign keys from column names
                table_schema = self.get_table_schema(table_name, schema)
                foreign_keys = []
                
                for column in table_schema:
                    col_name = column.get('column_name', '').lower()
                    if col_name.endswith('_id') and col_name != 'id' and col_name != f"{table_name.lower()}_id":
                        # Extract the potential referenced table
                        ref_table = col_name[:-3]  # Remove '_id'
                        
                        # Check if this table exists in our schema
                        all_tables, _ = self.get_all_tables(schema)
                        all_tables_lower = [t.lower() for t in all_tables]
                        if ref_table in all_tables_lower:
                            foreign_keys.append({
                                'column_name': column.get('column_name'),
                                'foreign_table': ref_table,
                                'foreign_column': f"{ref_table}_id"
                            })
            else:
                foreign_keys = []
                for _, row in df.iterrows():
                    foreign_keys.append({
                        'column_name': row['column_name'],
                        'foreign_table': row['foreign_table_name'],
                        'foreign_column': row['foreign_column_name']
                    })
            
            logger.info(f"Found {len(foreign_keys)} foreign keys for table {table_name}")
            
            # Cache and return the result
            self._foreign_keys_cache[cache_key] = foreign_keys
            return foreign_keys
        except Exception as e:
            logger.error(f"Error retrieving foreign keys for table {table_name}: {e}")
            self._foreign_keys_cache[cache_key] = []
            return []
    
    def get_all_schemas(self, schema: str = "events", tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get schemas for all tables or specified tables
        Uses caching to avoid frequent database calls
        """
        current_time = time.time()
        
        # Use cached schema if it's still valid and we're requesting all tables
        if (tables is None and 
            self.schema_cache and 
            current_time - self.cache_timestamp < self.cache_ttl):
            logger.info(f"Using cached schema info with {len(self.schema_cache)} tables")
            return self.schema_cache
        
        # If connection has failed before and we're requesting all tables,
        # return the empty schema info to avoid retrying everything
        if tables is None and self._db_connection_failed:
            logger.warning("Database connection previously failed, returning empty schema info")
            return {}
        
        schema_info = {}
        
        # Get all tables if not specified
        if tables is None:
            all_tables, found_schema = self.get_all_tables(schema)
            
            # Update schema to use what we found
            schema = found_schema if found_schema else schema
            
            # Limit the number of tables to process to avoid excessive queries
            if len(all_tables) > self.max_tables_to_process:
                logger.warning(f"Limiting schema query to {self.max_tables_to_process} tables (out of {len(all_tables)} total)")
                tables = all_tables[:self.max_tables_to_process]
            else:
                tables = all_tables
                
            # If no tables found, return empty dict
            if not tables:
                logger.warning("No tables found, returning empty schema info")
                return schema_info
        
        logger.info(f"Getting schema info for {len(tables)} tables in schema {schema}")
        
        # Get schema for each table
        for table_name in tables:
            try:
                # Skip tables we already know are empty
                if table_name in self._empty_tables:
                    logger.debug(f"Skipping known empty table {table_name}")
                    continue
                    
                table_schema = self.get_table_schema(table_name, schema)
                
                # If no columns were found, skip this table
                if not table_schema:
                    continue
                    
                primary_keys = self.get_table_primary_keys(table_name, schema)
                foreign_keys = self.get_table_foreign_keys(table_name, schema)
                
                schema_info[table_name] = {
                    'columns': table_schema,
                    'primary_keys': primary_keys,
                    'foreign_keys': foreign_keys
                }
            except Exception as e:
                logger.error(f"Error retrieving schema for table {table_name}: {e}")
                # Continue with other tables instead of failing completely
                continue
        
        logger.info(f"Retrieved schema information for {len(schema_info)} tables")
        
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
        tables, schema = self.get_all_tables()
        
        # Filter out empty tables
        tables = [t for t in tables if t not in self._empty_tables]
        
        relationships = {table: set() for table in tables}
        
        logger.info(f"Building relationship graph for {len(tables)} tables")
        
        for table in tables:
            foreign_keys = self.get_table_foreign_keys(table, schema)
            for fk in foreign_keys:
                foreign_table = fk['foreign_table']
                # Add relationship in both directions
                relationships[table].add(foreign_table)
                if foreign_table in relationships:
                    relationships[foreign_table].add(table)
        
        logger.info(f"Found relationships between {len(relationships)} tables")
        return relationships
        
    def clear_cache(self):
        """Clear all caches to force fresh data to be loaded"""
        self.schema_cache = {}
        self.cache_timestamp = 0
        self._tables_cache = None
        self._schema_name = None
        self._columns_cache = {}
        self._primary_keys_cache = {}
        self._foreign_keys_cache = {}
        self._empty_tables = set()
        self._db_connection_failed = False
        logger.info("Schema cache cleared")