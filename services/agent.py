import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from langgraph.graph import END, StateGraph
import copy

from core.config import settings
from core.database import RedshiftConnector
from services.llm_service import llm_service
from services.schema_service import SchemaService
from models.agent_state import AgentState
from models.conversation import conversation_store, Conversation, Message

logger = logging.getLogger(__name__)

class AgentService:
    """Service for managing the AI agent's workflow"""
    
    def __init__(self):
        """Initialize the agent service"""
        self.db_connector = RedshiftConnector()
        self.schema_service = SchemaService(self.db_connector)
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("understand_query", self.understand_query)
        workflow.add_node("ask_clarification", self.ask_for_clarification)
        workflow.add_node("generate_sql", self.generate_sql)
        workflow.add_node("query_db", self.query_database)
        workflow.add_node("formulate_response", self.formulate_response)
        
            # Define conditional routing with enhanced logging
        def route_after_understanding(state: AgentState):
            # Log the state details for debugging
            logger.info(f"[Routing] Current state needs_clarification: {state.needs_clarification}")
            logger.info(f"[Routing] Current query_understanding: {state.query_understanding}")
            
            # Ensure needs_clarification is checked from multiple sources
            needs_clarification = (
                state.needs_clarification or 
                (state.query_understanding and 
                state.query_understanding.get('needs_clarification', False)) or
                (state.query_understanding and 
                state.query_understanding.get('missing_info', []))
            )
            
            logger.info(f"[Routing] Determined needs_clarification: {needs_clarification}")
            
            return "ask_clarification" if needs_clarification else "generate_sql"
        
        # Add conditional edges using the newer API
        workflow.add_conditional_edges(
            "understand_query",
            route_after_understanding
        )
        
        # Add regular edges
        workflow.add_edge("ask_clarification", END)
        workflow.add_edge("generate_sql", "query_db")
        workflow.add_edge("query_db", "formulate_response")
        workflow.add_edge("formulate_response", END)
        
        # Set the entry point
        workflow.set_entry_point("understand_query")
        
        return workflow.compile()
    
    def understand_query(self, state: AgentState) -> AgentState:
        """Understand the user's query"""
        logger.info(f"[ServiceAgent] Starting to understand query: {state.current_query}")

        start_time = time.time()
        
        # Load table schemas if not already loaded
        try:
            if not state.table_schemas:
                state.table_schemas = self.schema_service.get_all_schemas()
                # If no schemas were returned, handle it gracefully
                if not state.table_schemas:
                    logger.warning("No table schemas were returned")
        except Exception as e:
            logger.error(f"Error loading table schemas: {str(e)}")
            state.error = f"Error connecting to the database: {str(e)}"
            state.final_response = f"I'm having trouble connecting to the database. Please check your connection settings and try again. Error details: {str(e)}"
            return state
        
        # Format table schemas for the LLM
        try:
            table_schemas_str = state.get_formatted_schemas()
        except Exception as e:
            logger.error(f"Error formatting schemas: {str(e)}")
            state.error = f"Error preparing database schema: {str(e)}"
            state.final_response = f"I encountered an issue while analyzing the database structure. Error details: {str(e)}"
            return state
        
        try:
            # Use LLM service to understand the query
            query_understanding = llm_service.understand_query(
                state.current_query, 
                table_schemas_str
            )

            logger.info(f"[ServiceAgent] Query understanding: {query_understanding}")
            
            state.query_understanding = query_understanding
            
            # Comprehensive check for needs_clarification
            state.needs_clarification = (
                query_understanding.get("needs_clarification", False) or
                bool(query_understanding.get("missing_info", []))
            )
            logger.info(f"[ServiceAgent] Setting needs_clarification to: {state.needs_clarification}")
            logger.info(f"[ServiceAgent] Query understanding missing_info: {query_understanding.get('missing_info', [])}")
            logger.info(f"[ServiceAgent] query understanding complete")

        except Exception as e:
            logger.error(f"[AgentService] Error understanding query: {str(e)}")
            state.error = f"Error understanding your query: {str(e)}"
            state.final_response = f"I'm having trouble understanding your request. This may be due to an issue with the AI service. Error details: {str(e)}"
        
        end_time = time.time()
        state.record_timing("understand_query", end_time - start_time)

        logger.info(f"[ServiceAgent] understand_query complete2. needs_clarification: {state.needs_clarification}")
        return state
    
    def ask_for_clarification(self, state: AgentState) -> AgentState:
        """Ask for clarification from the user"""
        logger.info(f"[ServiceAgent] Asking for clarification")
        start_time = time.time()
        
        try:
            missing_info = state.query_understanding.get("missing_info", [])
            if not missing_info:
                missing_info = ["additional details about your request"]
            
            # Use LLM service to generate clarification question
            clarification_question = llm_service.generate_clarification_question(
                state.current_query,
                missing_info
            )
            
            state.clarification_question = clarification_question
            
            # Add to conversation
            conversation = conversation_store.get_conversation(state.conversation_id)
            if conversation:
                conversation.add_message("assistant", clarification_question)
            
            logger.info(f"[ServiceAgent] Generated clarification question: {clarification_question}")
                
        except Exception as e:
            logger.error(f"Error generating clarification: {str(e)}")
            state.error = f"Error generating clarification: {str(e)}"
            state.clarification_question = "I need some more information to answer your question. Could you provide more details?"
            state.final_response = state.clarification_question + f" (Note: I encountered an error while processing your request: {str(e)})"
            
            # Add to conversation
            conversation = conversation_store.get_conversation(state.conversation_id)
            if conversation:
                conversation.add_message("assistant", state.final_response)
        
        end_time = time.time()
        state.record_timing("ask_clarification", end_time - start_time)
        
        return state
    
    def generate_sql(self, state: AgentState) -> AgentState:
        """Generate SQL query from the query understanding"""
        logger.info(f"[ServiceAgent] Generating SQL query")
        start_time = time.time()
        
        # If there was an error earlier, skip this step
        if state.error:
            return state
        
        # Format table schemas for the LLM
        table_schemas_str = state.get_formatted_schemas()
        
        try:
            # Use LLM service to generate SQL
            sql_query = llm_service.generate_sql(
                state.query_understanding,
                table_schemas_str
            )

             # Clean up the SQL query by removing code block markers
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()

            state.sql_query = sql_query

            logger.info(f"[ServiceAgent] Generated SQL query: {sql_query}")
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            state.error = f"Error generating SQL query: {str(e)}"
            state.final_response = f"I had trouble converting your request into a database query. Error details: {str(e)}"
        
        end_time = time.time()
        state.record_timing("generate_sql", end_time - start_time)
        
        return state
    
    def query_database(self, state: AgentState) -> AgentState:
        """Execute the SQL query against the database"""
        
        logger.info(f"[ServiceAgent] Querying database")
        start_time = time.time()
        
        # If there was an error earlier, skip this step
        if state.error:
            return state
        
        if not state.sql_query:
            state.error = "No SQL query to execute"
            state.final_response = "I couldn't generate a valid database query from your request. Please try rephrasing your question."
            return state
        
        try:
            # Validate the query before executing
            is_valid, error_message = self.db_connector.validate_sql(state.sql_query)
            if not is_valid:
                state.error = f"SQL validation error: {error_message}"
                logger.error(f"SQL validation error: {error_message}")
                state.final_response = f"The query I generated doesn't seem to be valid: {error_message}. Please try rephrasing your question."
                return state
            
            # Enforce query limits if configured
            if settings.AGENT_ENFORCE_LIMITS:
                max_rows = settings.REDSHIFT_MAX_ROWS
                limited_query = self.db_connector.enforce_query_limits(state.sql_query, max_rows)
                state.sql_query = limited_query
            
            # Execute the query
            df, columns = self.db_connector.execute_query(state.sql_query)
            state.sql_results = df
            
            # Log the execution in the conversation
            conversation = conversation_store.get_conversation(state.conversation_id)
            if conversation and settings.AGENT_LOG_QUERIES:
                row_count = len(df) if isinstance(df, pd.DataFrame) else 0
                execution_time = time.time() - start_time
                conversation.add_sql_execution(
                    state.sql_query, 
                    execution_time,
                    row_count
                )
            logger.info(f"[ServiceAgent] Query executed successfully, returned {row_count} rows")

        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            state.error = f"Error querying database: {str(e)}"
            state.sql_results = None
            state.final_response = f"I encountered an error when trying to execute the database query: {str(e)}. Please check your database connection settings."
            
            # Log the error in the conversation
            conversation = conversation_store.get_conversation(state.conversation_id)
            if conversation and settings.AGENT_LOG_QUERIES:
                execution_time = time.time() - start_time
                conversation.add_sql_execution(
                    state.sql_query, 
                    execution_time,
                    0,
                    str(e)
                )
        
        end_time = time.time()
        state.record_timing("query_database", end_time - start_time)
        
        return state
    
    def formulate_response(self, state: AgentState) -> AgentState:
        """Formulate a natural language response based on query results"""
        logger.info(f"[ServiceAgent] Formulating response")

        start_time = time.time()
        
        # If there's already a final response from error handling, don't overwrite it
        if state.final_response:
            logger.info(f"[ServiceAgent] Final response already set: {state.final_response}")
            # Add to conversation if not already added
            conversation = conversation_store.get_conversation(state.conversation_id)
            if conversation:
                conversation.add_message("assistant", state.final_response)
            return state
            
        if state.error:
            logger.error(f"Error encountered in previous steps: {state.error}")
            state.final_response = f"I encountered an error: {state.error}"
            
            # Add to conversation
            conversation = conversation_store.get_conversation(state.conversation_id)
            if conversation:
                conversation.add_message("assistant", state.final_response)
                
            end_time = time.time()
            state.record_timing("formulate_response", end_time - start_time)
            return state
        
         # If no results or empty DataFrame
        if (state.sql_results is None or 
            (isinstance(state.sql_results, pd.DataFrame) and len(state.sql_results) == 0)):
            state.final_response = "I didn't find any data matching your query. Could you try a different query or modify your search criteria?"
            
            logger.info(f"[ServiceAgent] No results found")
            # Add to conversation
            conversation = conversation_store.get_conversation(state.conversation_id)
            if conversation:
                conversation.add_message("assistant", state.final_response)
                
            end_time = time.time()
            state.record_timing("formulate_response", end_time - start_time)
            return state
        
        try:
            # Limit the results for the LLM context if it's a DataFrame
            sql_results_for_llm = state.sql_results
            if isinstance(sql_results_for_llm, pd.DataFrame):
                sql_results_for_llm = sql_results_for_llm.head(20)
            
            logger.info(f"[ServiceAgent] SQL results for LLM: {sql_results_for_llm}")
            # Use LLM service to formulate response
            response = llm_service.formulate_response(
                state.current_query,
                state.sql_query,
                sql_results_for_llm
            )
            
            state.final_response = response
            
            # Add to conversation
            conversation = conversation_store.get_conversation(state.conversation_id)
            if conversation:
                conversation.add_message("assistant", response)
                
        except Exception as e:
            logger.error(f"Error formulating response: {str(e)}")
            state.error = f"Error formulating response: {str(e)}"
            state.final_response = f"I found some results in the database, but encountered an error while analyzing them: {str(e)}. Please try simplifying your query."
            
            # Add to conversation
            conversation = conversation_store.get_conversation(state.conversation_id)
            if conversation:
                conversation.add_message("assistant", state.final_response)
        
        end_time = time.time()
        state.record_timing("formulate_response", end_time - start_time)
        
        logger.info(f"[ServiceAgent] Formulated response: {state.final_response}")
        return state
    
    def process_query(self, conversation_id: str, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a user query through the agent workflow
        
        Args:
            conversation_id: ID of the conversation
            query: User's natural language query
            
        Returns:
            Tuple of (response, context) where context contains additional information
        """
        logger.info(f"[ServiceAgent] Processing query for conversation: {conversation_id}")
        # Get or create conversation
        conversation = conversation_store.get_conversation(conversation_id)
        if not conversation:
            conversation = conversation_store.create_conversation()
            conversation_id = conversation.id
        
        # Add user message to conversation
        conversation.add_message("user", query)
        
        # Initialize state
        state = AgentState(conversation_id=conversation_id)
        state.current_query = query
        
        try:
            # Try to load table schemas - if this fails, we'll handle it
            try:
                state.table_schemas = self.schema_service.get_all_schemas()
                # Check if schemas were loaded successfully
                if not state.table_schemas:
                    logger.warning("No schemas were loaded, but no exception was thrown")
            except Exception as e:
                error_msg = f"I'm having trouble connecting to the database: {str(e)}. Please check your connection settings."
                conversation.add_message("assistant", error_msg)
                logger.error(f"Error loading table schemas: {str(e)}")
                return error_msg, {"error": str(e)}
            
            # Process through the graph
            result_values = self.graph.invoke(state)
            
            # The last state is always in the 'values' field with the key being the last node
            # Need to extract the actual final state value
            final_state = self._extract_final_state(result_values, state)
            
            logger.info(f"[ServiceAgent] Final state process query needs_clarification: {final_state.needs_clarification}")
            
            # Check if clarification is needed
            if final_state.needs_clarification:
                logger.info("[ServiceAgent] Clarification needed")
                # Use LLM service to generate a clarification question
                missing_info = final_state.query_understanding.get('missing_info', [])
                if not missing_info:
                    missing_info = ["additional details"]
                
                clarification_question = llm_service.generate_clarification_question(
                    query, 
                    missing_info
                )
                
                # Add clarification question to conversation
                conversation.add_message("assistant", clarification_question)
                
                return clarification_question, {
                    "needs_clarification": True,
                    "missing_info": missing_info
                }
            
            # Check if we have a final response, if not this is an error
            if not getattr(final_state, 'final_response', None):
                error_msg = "Sorry, I couldn't generate a valid response for your query."
                if getattr(final_state, 'error', None):
                    error_msg += f" Error details: {final_state.error}"
                final_state.final_response = error_msg
                conversation.add_message("assistant", error_msg)
            
            # Create context with additional information
            context = {
                "sql_query": getattr(final_state, 'sql_query', ''),
                "sql_results": getattr(final_state, 'sql_results', None),
                "timing": getattr(final_state, 'timing', {}),
                "needs_clarification": getattr(final_state, 'needs_clarification', False),
                "error": getattr(final_state, 'error', '')
            }

            logger.info(f"[ServiceAgent] Final state needs_clarification: {final_state.needs_clarification}")
            return final_state.final_response, context
                
        except Exception as e:
            logger.error(f"Error processing through graph: {str(e)}", exc_info=True)
            error_message = f"An error occurred while processing your query: {str(e)}"
            conversation.add_message("assistant", error_message)
            return error_message, {"error": str(e)}
    
    def _extract_final_state(self, result_values, original_state):
        """Helper method to extract the final state from LangGraph results"""
        logger.info(f"[ServiceAgent] Extracting final state")
        try:
            # Create a copy of the original state to preserve its attributes
            final_state = copy.deepcopy(original_state)

            logger.info(f"[ServiceAgent] Original state needs_clarification: {original_state.needs_clarification}")
            logger.info(f"[ServiceAgent] Original query_understanding: {original_state.query_understanding}")
            logger.info(f"[ServiceAgent] Result values type: {type(result_values)}")
            
            # Try to extract state from different possible structures
            extracted_state = None
            
            # Case 1: If result_values has 'values' attribute
            if hasattr(result_values, 'values'):
                # If values is a dictionary
                if isinstance(result_values.values, dict):
                    # Find the last state object in the values
                    for value in reversed(list(result_values.values.values())):
                        # Handle different potential types of value
                        if isinstance(value, (dict, AgentState)):
                            extracted_state = value
                            break
                        elif hasattr(value, '__dict__'):
                            extracted_state = value.__dict__
                            break
            
            # Case 2: If result_values itself is a dictionary
            if extracted_state is None and isinstance(result_values, dict):
                extracted_state = result_values
            
            # Merge extracted state if found
            if extracted_state:
                # Safely handle various types of attributes
                for attr, val in extracted_state.items():
                    try:
                        # Skip None or empty values, but be careful with DataFrame
                        if val is not None:
                            # Special handling for DataFrame
                            if hasattr(val, 'empty'):
                                if not val.empty:
                                    setattr(final_state, attr, val)
                            elif val != '':
                                setattr(final_state, attr, val)
                    except Exception as attr_error:
                        logger.warning(f"Error setting attribute {attr}: {attr_error}")
            
            # Preserve crucial original state attributes
            final_state.conversation_id = original_state.conversation_id
            
            # Preserve needs_clarification 
            final_state.needs_clarification = (
                original_state.needs_clarification or 
                (original_state.query_understanding and 
                original_state.query_understanding.get('needs_clarification', False)) or
                (hasattr(final_state, 'query_understanding') and 
                final_state.query_understanding.get('needs_clarification', False))
            )

            # Ensure final_response is set for zero-result scenarios
            if (getattr(final_state, 'sql_results', None) is not None and 
                hasattr(final_state.sql_results, 'empty') and 
                final_state.sql_results.empty and 
                not getattr(final_state, 'final_response', None)):
                final_state.final_response = "I didn't find any data matching your query. Could you try a different search criteria?"

            logger.info(f"[ServiceAgent] Final state needs_clarification: {final_state.needs_clarification}")
            
            return final_state
        
        except Exception as e:
            logger.error(f"Error extracting final state: {str(e)}", exc_info=True)
            
            # Enhanced fallback
            fallback_state = copy.deepcopy(original_state)
            fallback_state.error = str(e)
            fallback_state.final_response = f"An error occurred while processing your query: {str(e)}"
            
            return fallback_state
        
    def process_clarification(self, conversation_id: str, clarification_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a clarification response from the user
        
        Args:
            conversation_id: ID of the conversation
            clarification_input: User's response to the clarification question
            
        Returns:
            Tuple of (response, context) similar to process_query
        """
        logger.info(f"[ServiceAgent] Processing clarification for conversation {conversation_id}: {clarification_input}")
        # Get conversation
        conversation = conversation_store.get_conversation(conversation_id)
        if not conversation:
            # This shouldn't happen in normal flow
            logger.error(f"Conversation {conversation_id} not found for clarification")
            conversation = conversation_store.create_conversation()
            conversation_id = conversation.id
        
        # Add user message to conversation
        conversation.add_message("user", clarification_input)
        
        # Get the last query - combine with clarification
        user_messages = [msg for msg in conversation.messages if msg.role == "user"]
        if len(user_messages) >= 2:
            original_query = user_messages[-2].content
            combined_query = f"{original_query} {clarification_input}"
        else:
            combined_query = clarification_input
        
        # Initialize state
        state = AgentState(conversation_id=conversation_id)
        state.current_query = combined_query
        
        try:
            # Try to load table schemas - if this fails, we'll handle it
            try:
                state.table_schemas = self.schema_service.get_all_schemas()
                # Check if schemas were loaded successfully
                if not state.table_schemas:
                    logger.warning("No schemas were loaded, but no exception was thrown")
            except Exception as e:
                error_msg = f"I'm having trouble connecting to the database: {str(e)}. Please check your connection settings."
                conversation.add_message("assistant", error_msg)
                return error_msg, {"error": str(e)}
                
            # Process through the graph
            result_values = self.graph.invoke(state)
            
            # Extract the final state
            final_state = self._extract_final_state(result_values, state)
            
            # Check if we have a final response, if not this is an error
            if not getattr(final_state, 'final_response', None) and not getattr(final_state, 'clarification_question', None):
                error_msg = "Sorry, I encountered an issue processing your query and couldn't generate a valid response."
                if getattr(final_state, 'error', None):
                    error_msg += f" Error details: {final_state.error}"
                final_state.final_response = error_msg
                conversation.add_message("assistant", error_msg)
            
            # Create context with additional information
            context = {
                "sql_query": getattr(final_state, 'sql_query', ''),
                "sql_results": getattr(final_state, 'sql_results', None),
                "timing": getattr(final_state, 'timing', {}),
                "needs_clarification": getattr(final_state, 'needs_clarification', False),
                "error": getattr(final_state, 'error', '')
            }
            
            if context["needs_clarification"]:
                return getattr(final_state, 'clarification_question', 'Could you provide more details?'), context
            else:
                return getattr(final_state, 'final_response', 'No response generated'), context
        
        except Exception as e:
            logger.error(f"Error processing clarification: {str(e)}", exc_info=True)
            error_message = f"An error occurred while processing your clarification: {str(e)}"
            conversation.add_message("assistant", error_message)
            return error_message, {"error": str(e)}


# Create a global instance
agent_service = AgentService()