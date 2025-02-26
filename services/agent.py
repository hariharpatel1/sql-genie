from datetime import date, datetime
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from langgraph.graph import END, StateGraph
import copy
import re
from IPython.display import Image, display

from core.config import settings
from core.database import RedshiftConnector
from services.llm_data_query_service import llm_data_query_service
from services.llm_service import llm_service
from services.schema_service import SchemaService
from services.data_context_service import data_context_service
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
        workflow.add_node("check_context", self.check_context)  # Added check_context node
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
            
            return "ask_clarification" if needs_clarification else "check_context"  # Route to check_context instead of generate_sql
        
        # Define routing after context check
        def route_after_context_check(state: AgentState):
            logger.info(f"[Routing] Current state use in memory: {state.use_in_memory_data}")
            # Check if we can use in-memory data
            if state.can_answer_from_memory and state.use_in_memory_data:
                logger.info(f"[Routing] Using in-memory data, skipping SQL generation")
                return "formulate_response"
            else:
                return "generate_sql"
        
        # Add conditional edges using the newer API
        workflow.add_conditional_edges(
            "understand_query",
            route_after_understanding
        )
        
        # Add conditional edge from check_context
        workflow.add_conditional_edges(
            "check_context",
            route_after_context_check
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

        try:
            display(Image(graph.get_graph().draw_mermaid_png()))
        except Exception:
            # This requires some extra dependencies and is optional
            pass

        
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
            
            # Add context summary to help the LLM understand what data is already available
            context_summary = data_context_service.get_context_summary(state.conversation_id)
            
            # Add available in-memory tables to the context
            if context_summary.get('tables'):
                table_schemas_str += "\n\nIn-memory tables from previous queries:\n"
                for table in context_summary.get('tables', []):
                    table_schemas_str += f"Table: {table['name']} ({table['rows']} rows)\n"
                    table_schemas_str += f"Columns: {', '.join(table['column_names'])}\n"
                    if table.get('sample'):
                         table_schemas_str += f"Sample: {json.dumps(table['sample'], default=self.custom_json_serializer, indent=2)[:200]}...\n\n"

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
            
            # Add a new field to detect data enrichment requests
            if "add_data" in state.current_query.lower() or "enrich" in state.current_query.lower():
                state.query_understanding["data_enrichment_request"] = True
                
            logger.info(f"[ServiceAgent] Setting needs_clarification to: {state.needs_clarification}")
            logger.info(f"[ServiceAgent] Query understanding missing_info: {query_understanding.get('missing_info', [])}")
            logger.info(f"[ServiceAgent] query understanding complete")

        except Exception as e:
            logger.error(f"[AgentService] Error understanding query: {str(e)}")
            state.error = f"Error understanding your query: {str(e)}"
            state.final_response = f"I'm having trouble understanding your request. This may be due to an issue with the AI service. Error details: {str(e)}"
        
        end_time = time.time()
        state.record_timing("understand_query", end_time - start_time)

        logger.info(f"[ServiceAgent] understand_query complete. needs_clarification: {state.needs_clarification}")
        return state
    
    def ask_for_clarification(self, state: AgentState) -> AgentState:
        """Ask for clarification from the user"""
        logger.info(f"[ServiceAgent] Asking for clarification")
        start_time = time.time()
        
        try:
            missing_info = state.query_understanding.get("missing_info", [])
            if not missing_info:
                missing_info = ["additional details about your request"]
            
            # Check if this is a data enrichment request
            if state.query_understanding.get("data_enrichment_request", False):
                clarification_question = "I'd be happy to add that data to our conversation. Could you please provide the information in a structured format? You can share it as a table, CSV, or simply list the data points clearly."
            else:
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
    
    def check_context(self, state: AgentState) -> AgentState:
        """Enhanced context checking that prioritizes existing conversation data"""
        logger.info(f"[ServiceAgent] Comprehensive context checking for query: {state.current_query}")
        start_time = time.time()

        # Check if in-memory data usage is allowed
        if not getattr(state, 'use_memory_allowed', True):
            logger.info("[ServiceAgent] In-memory data usage is disabled")
            # Skip context checking and proceed to SQL generation
            end_time = time.time()
            state.record_timing("check_context", end_time - start_time)
            return state

        # Get full context summary
        context_summary = data_context_service.get_context_summary(state.conversation_id)
        logger.info(f"[ServiceAgent] Context summary: {context_summary}")

        # Safely serialize context summary
        try:
            serialized_context = json.dumps(context_summary, default=self.custom_json_serializer, indent=2)
        except Exception as serialize_error:
            logger.error(f"Error serializing context summary: {serialize_error}")
            serialized_context = str(context_summary)

        # Prepare a comprehensive context for LLM analysis
        comprehensive_prompt = f"""
        Conversation Context Summary:
        {serialized_context}

        User Query: "{state.current_query}"

        Analyze the user query in relation to the existing conversation context:
        1. identify if the query is for a new data request or a follow-up
        2. if this is a follow-up, can answer from context should alsways be true
        3. Can this query be answered using existing data?
        4. If not, what type of additional data or action is needed?
        5. Determine if this requires:
            a) In-memory data analysis
            b) SQL query generation
            c) Clarification from user
            d) External data enrichment

        Respond in structured JSON:
        {{
            "can_answer_from_context": true/false,
            "recommended_action": "in_memory_analysis"/"generate_sql"/"ask_clarification"/"data_enrichment",
            "reasoning": "Explanation of the recommendation",
            "relevant_data_sources": ["table1", "table2"]
        }}
        """

        try:
            # Use LLM to analyze the context and query
            context_analysis = llm_service.generate_text(comprehensive_prompt)
            logger.info(f"[ServiceAgent] Context analysis result: {context_analysis}")

            # Parse the JSON response
            try:
                analysis_result = json.loads(context_analysis)
            except json.JSONDecodeError:
                # Fallback parsing
                json_match = re.search(r'\{.*\}', context_analysis, re.DOTALL)
                analysis_result = json.loads(json_match.group(0)) if json_match else {}

            state.can_answer_from_memory = analysis_result.get('can_answer_from_context', False)

            # Determine next steps based on LLM analysis
            if analysis_result.get('can_answer_from_context', False):
                # Use in-memory data analysis
                result_df, response_text = llm_data_query_service.execute_in_memory_query(
                    state.conversation_id,
                    state.current_query
                )
            
                state.in_memory_results = result_df
                state.final_response = response_text
                logger.info("[ServiceAgent] Answered from existing context")

            elif analysis_result.get('recommended_action') == 'ask_clarification':
                state.needs_clarification = True
                state.clarification_question = analysis_result.get('reasoning', 'I need more information to proceed.')

            elif analysis_result.get('recommended_action') == 'data_enrichment':
                state.needs_clarification = True
                state.clarification_question = "I need additional data to fully answer your query. Could you provide more details or context?"

            # If no clear resolution, proceed to SQL generation
            

        except Exception as e:
            logger.error(f"Error in comprehensive context checking: {e}")
            # Fallback to existing logic

        end_time = time.time()
        state.record_timing("check_context", end_time - start_time)
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
            
            # Enhance join queries using LLM
            from services.llm_data_query_service import llm_data_query_service
            
            sql_query = llm_data_query_service.enhance_join_query(sql_query, state.table_schemas)
            
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
            
            # Enrich context with query results
            self._enrich_context(state)
            
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
            
            row_count = len(df) if isinstance(df, pd.DataFrame) else 0
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
    
    def _enrich_context(self, state: AgentState) -> None:
        """Store query results in the conversation context"""
        logger.info(f"[ServiceAgent] Enriching conversation context")

        # Skip context enrichment if in-memory data is not used
        if not state.use_in_memory_data:
            return
        
        # If there was an error or no results, skip this step
        if state.error or state.sql_results is None:
            return
        
        try:
            # Store the query results in the conversation context
            if isinstance(state.sql_results, pd.DataFrame) and not state.sql_results.empty:
                df_id = data_context_service.store_query_result(
                    state.conversation_id,
                    state.sql_results,
                    state.sql_query
                )
                
                # Try to extract a meaningful table name from the query
                table_name = self._extract_main_table_name(state.sql_query)
                if table_name:
                    # Register the DataFrame as a named table
                    data_context_service.register_table(
                        state.conversation_id,
                        table_name,
                        df_id
                    )
                
                logger.info(f"[ServiceAgent] Stored query results in context, df_id: {df_id}")
            
        except Exception as e:
            logger.error(f"Error enriching context: {str(e)}")
            # Don't set an error state, this is a non-critical failure
    
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
            
            # For now, use the first table as the main one
            main_table = data_context_service._extract_main_table_name(sql_query)
            
            # Generate a result name based on the query type
            if "SELECT" in sql_query.upper():
                if len(table_names) > 1:
                    return f"{main_table}_joined_data"
                else:
                    return f"{main_table}_results"
            
            return main_table
            
        except Exception as e:
            logger.error(f"Error extracting table name: {str(e)}")
            return "query_results"
    
    def formulate_response(self, state: AgentState) -> AgentState:
        """Formulate a natural language response based on query results"""
        logger.info(f"[ServiceAgent] Formulating response")

        start_time = time.time()
        
        # If there's already a final response from error handling or in-memory processing, don't overwrite it
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
            # For in-memory data, we've already set the response in check_context
            if state.use_in_memory_data and state.final_response:
                # Just ensure it's added to the conversation
                conversation = conversation_store.get_conversation(state.conversation_id)
                if conversation:
                    conversation.add_message("assistant", state.final_response)
                
                end_time = time.time()
                state.record_timing("formulate_response", end_time - start_time)
                return state
            
            # Create enhanced context with both the database results and in-memory data
            enhanced_response = llm_data_query_service.answer_with_context(
                state.conversation_id,
                state.current_query,
                {
                    "database_query": state.sql_query,
                    "database_results": state.sql_results.head(20).to_dict(orient='records') 
                        if isinstance(state.sql_results, pd.DataFrame) else state.sql_results,
                    "row_count": len(state.sql_results) if isinstance(state.sql_results, pd.DataFrame) else 0
                }
            )
            
            state.final_response = enhanced_response
            
            # Add to conversation
            conversation = conversation_store.get_conversation(state.conversation_id)
            if conversation:
                conversation.add_message("assistant", enhanced_response)
                
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
            
            # Preserve use_in_memory_data flag
            if hasattr(original_state, 'use_in_memory_data'):
                final_state.use_in_memory_data = original_state.use_in_memory_data
                final_state.can_answer_from_memory = getattr(original_state, 'can_answer_from_memory', False)

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
    
    def process_query(self, conversation_id: str, query: str, use_memory: bool = True) -> Tuple[str, Dict[str, Any]]:
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
            conversation = conversation_store.create_conversation(conversation_id)
            conversation_id = conversation.id
        
        # Add user message to conversation
        conversation.add_message("user", query)
        
        # Initialize state
        state = AgentState(conversation_id=conversation_id)
        state.current_query = query
        state.use_in_memory_data = use_memory
        state.can_answer_from_memory = False
        
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
                "error": getattr(final_state, 'error', ''),
                "use_in_memory_data": getattr(final_state, 'use_in_memory_data', False)
            }

            logger.info(f"[ServiceAgent] Final state needs_clarification: {final_state.needs_clarification}")
            return final_state.final_response, context
                
        except Exception as e:
            logger.error(f"Error processing through graph: {str(e)}", exc_info=True)
            error_message = f"An error occurred while processing your query: {str(e)}"
            conversation.add_message("assistant", error_message)
            return error_message, {"error": str(e)}
        
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
            conversation = conversation_store.create_conversation(conversation_id)
            conversation_id = conversation.id
        
        # Add user message to conversation
        conversation.add_message("user", clarification_input)
        
        # Check if this might be data enrichment
        if self._is_data_enrichment_input(clarification_input):
            try:
                # Try to parse the user input as data
                data = self._parse_user_data_input(clarification_input)
                if data is not None:
                    # Store the user data
                    success = self.process_user_data_enrichment(
                        conversation_id,
                        data,
                        "User provided data"
                    )
                    
                    if success:
                        response = "I've successfully added your data to our conversation. I'll use this information to answer your future questions. Is there anything specific you'd like to know about this data?"
                        conversation.add_message("assistant", response)
                        return response, {"data_enrichment": True}
            except Exception as e:
                logger.error(f"Error processing potential data enrichment: {str(e)}")
                # Continue with regular clarification flow
        
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
                "error": getattr(final_state, 'error', ''),
                "use_in_memory_data": getattr(final_state, 'use_in_memory_data', False)
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
    
    def _is_data_enrichment_input(self, user_input: str) -> bool:
        """Check if the user input appears to be data enrichment"""
        # Look for patterns that suggest structured data
        data_patterns = [
            r'\{\s*"[^"]+"\s*:',  # JSON-like
            r'\[\s*\{\s*"[^"]+"\s*:',  # JSON array
            r'\w+\s*,\s*\w+\s*,\s*\w+',  # CSV-like
            r'\|\s*\w+\s*\|\s*\w+\s*\|',  # Markdown table
        ]
        
        for pattern in data_patterns:
            if re.search(pattern, user_input):
                return True
        
        return False
    
    def _parse_user_data_input(self, user_input: str) -> Optional[Any]:
        """
        Try to parse user input as structured data
        
        Returns:
            DataFrame, list of dicts, or None if parsing failed
        """
        # Try to parse as JSON
        try:
            # Check for JSON array or object
            if (user_input.strip().startswith('[') and user_input.strip().endswith(']')) or \
               (user_input.strip().startswith('{') and user_input.strip().endswith('}')):
                import json
                data = json.loads(user_input)
                
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    return pd.DataFrame([data])
                return data
        except json.JSONDecodeError:
            pass
        
        # Try to parse as CSV
        try:
            if ',' in user_input:
                lines = user_input.strip().split('\n')
                if len(lines) >= 2:  # Need at least header and one data row
                    import io
                    return pd.read_csv(io.StringIO(user_input))
        except Exception:
            pass
        
        # Try to parse as markdown table
        try:
            if '|' in user_input:
                lines = user_input.strip().split('\n')
                if len(lines) >= 3 and all('|' in line for line in lines):
                    # Extract header
                    header_line = lines[0]
                    headers = [h.strip() for h in header_line.split('|') if h.strip()]
                    
                    # Skip separator line
                    data_lines = lines[2:] if '---' in lines[1] else lines[1:]
                    
                    # Parse data rows
                    data = []
                    for line in data_lines:
                        if not line.strip() or '|' not in line:
                            continue
                        values = [v.strip() for v in line.split('|') if v.strip()]
                        if len(values) == len(headers):
                            row = {headers[i]: values[i] for i in range(len(headers))}
                            data.append(row)
                    
                    if data:
                        return pd.DataFrame(data)
        except Exception:
            pass
        
        # Could not parse as structured data
        return None
    
    def process_user_data_enrichment(self, conversation_id: str, data: Any, description: str) -> bool:
        """
        Process user-provided data to enrich the conversation context
        
        Args:
            conversation_id: ID of the conversation
            data: User data to add to the context
            description: Description of the data
            
        Returns:
            Success status
        """
        try:
            # Convert data to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, dict):
                    df = pd.DataFrame([data])
                elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    df = pd.DataFrame(data)
                else:
                    logger.error(f"Invalid data format for enrichment: {type(data)}")
                    return False
            else:
                df = data
            
            # Store in context
            df_id = data_context_service.store_user_data(
                conversation_id,
                df,
                description
            )
            
            # Try to extract a meaningful table name
            table_name = f"user_{description.lower().replace(' ', '_')}"
            
            # Register the DataFrame as a named table
            data_context_service.register_table(
                conversation_id,
                table_name,
                df_id
            )
            
            logger.info(f"Successfully enriched context with user data, df_id: {df_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing user data enrichment: {str(e)}")
            return False
        
    def custom_json_serializer(self, obj):
        """Convert non-serializable objects to serializable format"""
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):  # For DataFrame objects
            return obj.to_dict()
        elif hasattr(obj, 'tolist'):  # For numpy arrays
            return obj.tolist()
        return str(obj)


# Create a global instance
agent_service = AgentService()