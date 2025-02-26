import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import logging
import uuid
import traceback
import json
from typing import Dict, Any, Optional
import base64
from io import BytesIO

from core.config import settings
from core.database import RedshiftConnector
from services.agent import agent_service
from services.data_context_service import data_context_service
from models.conversation import conversation_store
from ui.visualization import create_visualization, create_advanced_visualization

logger = logging.getLogger(__name__)

class StreamlitUI:
    """UI component for the Streamlit application"""
    
    def __init__(self):
        """Initialize the UI"""
        self.db_connector = RedshiftConnector()
        self.initialize_session_state()
        self.configure_page()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
       # Check if conversation_id exists, if not create a new one
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        
        # Ensure conversation exists in the conversation store
        conversation = conversation_store.get_conversation(st.session_state.conversation_id)
        if not conversation:
            conversation = conversation_store.create_conversation(st.session_state.conversation_id)
            
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "waiting_for_clarification" not in st.session_state:
            st.session_state.waiting_for_clarification = False
        
        if "current_response" not in st.session_state:
            st.session_state.current_response = None
        
        if "current_context" not in st.session_state:
            st.session_state.current_context = {}
            
        if "connection_status" not in st.session_state:
            st.session_state.connection_status = None
            
        if "db_error" not in st.session_state:
            st.session_state.db_error = None
            
        if "llm_status" not in st.session_state:
            st.session_state.llm_status = None
            
        if "data_enrichment_mode" not in st.session_state:
            st.session_state.data_enrichment_mode = False
            
        if "in_memory_data" not in st.session_state:
            st.session_state.in_memory_data = {}
    
    def configure_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="SQL Genie 🧞‍♂️",
            page_icon="🧞‍♂️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #4B3BC7;
        }
        .sub-header {
            color: #6E56CF;
        }
        .stApp {
            background-color: #F8F9FA;
        }
        .data-enrichment-container {
            background-color: #F0F7FF;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #BDD5FF;
        }
        .memory-data-box {
            background-color: #F8F0FF;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #E0C8FF;
            margin: 10px 0;
        }
        .download-button {
            display: inline-block;
            padding: 8px 16px;
            background-color: #4B3BC7;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin: 10px 0;
            font-weight: bold;
        }
        .download-button:hover {
            background-color: #3E30A0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def setup_sidebar(self):
        """Set up the sidebar with controls and status information"""
        with st.sidebar:
            st.header("🧞 Settings")
            
            # Database connection status
            st.subheader("🔌 Database Connection")
            db_status_container = st.container()
            
            # Test connection button
            if st.button("Test Connection"):
                with db_status_container:
                    with st.spinner("Testing connection..."):
                        try:
                            if self.db_connector.test_connection():
                                st.session_state.connection_status = "connected"
                                st.session_state.db_error = None
                                st.success("Connected to Redshift! ✅")
                            else:
                                st.session_state.connection_status = "failed"
                                st.session_state.db_error = "Failed to connect to Redshift"
                                st.error("Failed to connect to Redshift ❌")
                        except Exception as e:
                            st.session_state.connection_status = "error"
                            st.session_state.db_error = str(e)
                            st.error(f"Error connecting to Redshift: {str(e)} ❌")
            
            # Display the last connection status
            if st.session_state.connection_status == "connected":
                db_status_container.success("Connected to Redshift! ✅")
            elif st.session_state.connection_status == "failed":
                db_status_container.error("Failed to connect to Redshift ❌")
            elif st.session_state.connection_status == "error":
                db_status_container.error(f"Error: {st.session_state.db_error} ❌")
                
            # Show connection settings if there are problems
            if st.session_state.connection_status in ["failed", "error"]:
                with st.expander("Connection Settings"):
                    st.info("""
                    Check your .env file for these settings:
                    - REDSHIFT_DBNAME
                    - REDSHIFT_USER
                    - REDSHIFT_PASSWORD
                    - REDSHIFT_HOST
                    - REDSHIFT_PORT
                    """)
                    
                    st.warning("If you're running in demo mode without a real database, some features may not work.")
            
            # Show LLM settings status
            st.subheader("🤖 LLM Configuration")
            llm_status_container = st.container()
            
            # Try to see if Azure OpenAI settings are configured
            if (hasattr(settings, 'AZURE_OPENAI_API_KEY') and 
                hasattr(settings, 'AZURE_OPENAI_ENDPOINT') and 
                hasattr(settings, 'AZURE_OPENAI_DEPLOYMENT_NAME') and
                settings.AZURE_OPENAI_API_KEY and 
                settings.AZURE_OPENAI_ENDPOINT and 
                settings.AZURE_OPENAI_DEPLOYMENT_NAME):
                llm_status_container.success("Azure OpenAI settings configured ✅")
                st.session_state.llm_status = "configured"
            else:
                llm_status_container.warning("Azure OpenAI settings not fully configured ⚠️")
                st.session_state.llm_status = "incomplete"
                with st.expander("LLM Settings"):
                    st.info("""
                    Check your .env file for these settings:
                    - AZURE_OPENAI_API_KEY
                    - AZURE_OPENAI_ENDPOINT
                    - AZURE_OPENAI_API_VERSION
                    - AZURE_OPENAI_DEPLOYMENT_NAME
                    """)
            
            # New data enrichment section
            st.subheader("📊 Data Context")
            
            # Get current in-memory tables
            context_summary = data_context_service.get_context_summary(st.session_state.conversation_id)
            tables = context_summary.get('tables', [])
            
            if tables:
                st.success(f"{len(tables)} tables in memory ✅")
                with st.expander("In-Memory Data"):
                    for table in tables:
                        st.markdown(f"**{table['name']}** ({table['rows']} rows)")
                        st.dataframe(pd.DataFrame(table['sample']), height=150)
            else:
                st.info("No data in memory yet")
            
            # Add data enrichment button
            if st.button("🔍 Add Your Own Data"):
                st.session_state.data_enrichment_mode = not st.session_state.data_enrichment_mode
            
            # UI options
            st.subheader("🎨 Display Options")
            show_sql = st.checkbox("Show SQL Queries", value=True)
            show_timing = st.checkbox("Show Timing Information", value=False)
            use_memory = st.checkbox("Use In-Memory Data When Possible", value=False)
            
            # Update session state
            st.session_state.show_sql = show_sql
            st.session_state.show_timing = show_timing
            st.session_state.use_memory = use_memory
        
            
            # New conversation button
            st.subheader("💬 Conversation")
            if st.button("New Conversation 🔄"):
                self.start_new_conversation()
            
            # About section
            st.markdown("---")
            st.markdown("**SQL Genie 🧞‍♂️**")
            st.markdown("Ask questions about your data in natural language, with conversation context persistence.")
    
    def start_new_conversation(self):
        # Clear the old conversation from data context service
        data_context_service.clear_context(st.session_state.conversation_id)
        
        # Generate a truly new conversation ID
        st.session_state.conversation_id = str(uuid.uuid4())
        
        # Create a new conversation with this ID
        conversation_store.create_conversation(st.session_state.conversation_id)
        
        # Reset other session state variables
        st.session_state.messages = []
        st.session_state.waiting_for_clarification = False
        st.session_state.current_response = None
        st.session_state.current_context = {}
        st.session_state.data_enrichment_mode = False
        st.session_state.in_memory_data = {}

    
    def display_messages(self):
        """Display chat messages from history"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def generate_csv_download_link(self, df):
        """Generate a download link for a DataFrame as CSV"""
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="query_results.csv" class="download-button">📥 Download CSV</a>'
        return href
    
    def show_data_enrichment_panel(self):
        """Show the data enrichment input panel"""
        with st.container():
            st.markdown('<div class="data-enrichment-container">', unsafe_allow_html=True)
            st.subheader("📊 Add Your Own Data")
            
            # Options for data input
            data_input_method = st.radio(
                "How would you like to add data?",
                ["JSON", "CSV", "Table Editor", "Cancel"],
                horizontal=True
            )
            
            if data_input_method == "JSON":
                json_input = st.text_area(
                    "Paste your JSON data here:",
                    height=200,
                    help="Provide a JSON object or array of objects"
                )
                
                if st.button("Process JSON Data"):
                    if json_input:
                        try:
                            # Parse JSON and add to context
                            json_data = json.loads(json_input)
                            
                            # Provide a description
                            description = st.text_input("Provide a name for this dataset:", "user_data")
                            
                            # Add to context
                            success = agent_service.process_user_data_enrichment(
                                st.session_state.conversation_id,
                                json_data,
                                description
                            )
                            
                            if success:
                                st.success("Data successfully added to conversation context!")
                                st.session_state.data_enrichment_mode = False
                                # Add a virtual message to the conversation
                                system_msg = f"I've added your {description} data to our conversation context. You can now ask questions about it."
                                st.session_state.messages.append({"role": "assistant", "content": system_msg})
                            else:
                                st.error("Failed to add data to context")
                        except json.JSONDecodeError:
                            st.error("Invalid JSON format. Please check your input.")
                        except Exception as e:
                            st.error(f"Error processing data: {str(e)}")
            
            elif data_input_method == "CSV":
                csv_input = st.text_area(
                    "Paste your CSV data here:",
                    height=200,
                    help="Comma-separated values with a header row"
                )
                
                if st.button("Process CSV Data"):
                    if csv_input:
                        try:
                            # Parse CSV and add to context
                            import io
                            csv_data = pd.read_csv(io.StringIO(csv_input))
                            
                            # Provide a description
                            description = st.text_input("Provide a name for this dataset:", "user_csv_data")
                            
                            # Add to context
                            success = agent_service.process_user_data_enrichment(
                                st.session_state.conversation_id,
                                csv_data,
                                description
                            )
                            
                            if success:
                                st.success("CSV data successfully added to conversation context!")
                                st.session_state.data_enrichment_mode = False
                                # Add a virtual message to the conversation
                                system_msg = f"I've added your {description} data to our conversation context. You can now ask questions about it."
                                st.session_state.messages.append({"role": "assistant", "content": system_msg})
                            else:
                                st.error("Failed to add data to context")
                        except Exception as e:
                            st.error(f"Error processing CSV data: {str(e)}")
            
            elif data_input_method == "Table Editor":
                # Create an empty DataFrame for editing
                if "editor_data" not in st.session_state:
                    st.session_state.editor_data = pd.DataFrame({
                        'Column1': ['Value1', 'Value2'],
                        'Column2': [10, 20]
                    })
                
                # Allow user to edit the table
                edited_df = st.data_editor(
                    st.session_state.editor_data,
                    num_rows="dynamic",
                    use_container_width=True
                )
                
                # Store the edited data back in session state
                st.session_state.editor_data = edited_df
                
                if st.button("Add Table Data"):
                    if not edited_df.empty:
                        try:
                            # Provide a description
                            description = st.text_input("Provide a name for this dataset:", "user_table_data")
                            
                            # Add to context
                            success = agent_service.process_user_data_enrichment(
                                st.session_state.conversation_id,
                                edited_df,
                                description
                            )
                            
                            if success:
                                st.success("Table data successfully added to conversation context!")
                                st.session_state.data_enrichment_mode = False
                                # Add a virtual message to the conversation
                                system_msg = f"I've added your {description} data to our conversation context. You can now ask questions about it."
                                st.session_state.messages.append({"role": "assistant", "content": system_msg})
                            else:
                                st.error("Failed to add data to context")
                        except Exception as e:
                            st.error(f"Error processing table data: {str(e)}")
            
            elif data_input_method == "Cancel":
                st.session_state.data_enrichment_mode = False
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def handle_user_input(self, user_input: str):
        """Handle user input and generate response"""
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
       # Show thinking indicator with rotating circle progress
        thinking_placeholder = st.empty()
        with thinking_placeholder.container():
            with st.chat_message("assistant", avatar="🧞‍♂️"):
                # Use CSS to create a rotating circle progress
                st.markdown("""
                <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                .spinner {
                    display: inline-block;
                    border: 3px solid rgba(0, 0, 0, 0.1);
                    border-radius: 50%;
                    border-top: 3px solid #3498db;
                    width: 20px;
                    height: 20px;
                    animation: spin 1s linear infinite;
                    margin-left: 10px;
                    vertical-align: middle;
                }
                </style>
                <div>
                    <span style="vertical-align: middle;">Thinking...💭</span>
                    <div class="spinner"></div>
                </div>
                """, unsafe_allow_html=True)
        
        try:
            # Pre-check for configuration issues
            if st.session_state.llm_status == "incomplete":
                thinking_placeholder.empty()
                error_message = "LLM configuration is incomplete. Please check your Azure OpenAI settings in the sidebar."
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                    st.error(error_message)
                return
            
            # Modify query processing to respect use_memory setting
            query_kwargs = {
                "use_memory": st.session_state.get('use_memory', False)
            }
            
            # Process input based on state
            if st.session_state.waiting_for_clarification:
                # Handle clarification response
                response, context = agent_service.process_clarification(
                    st.session_state.conversation_id,
                    user_input
                )
                st.session_state.waiting_for_clarification = False
            else:
                # Handle new query
                response, context = agent_service.process_query(
                    st.session_state.conversation_id,
                    user_input,
                    **query_kwargs
                )
            
            # Save response and context
            st.session_state.current_response = response
            st.session_state.current_context = context
            
            # Clear thinking indicator
            thinking_placeholder.empty()
            
            # Display response
            self.display_assistant_response(response, context)
            
            # Check if we need clarification for the next round
            if context.get("needs_clarification", False):
                st.session_state.waiting_for_clarification = True
                
        except Exception as e:
            # Clear thinking indicator
            thinking_placeholder.empty()
            
            # Display error message
            error_message = f"An error occurred while processing your request: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            
            with st.chat_message("assistant"):
                st.error(error_message)
                
                # Show technical details in debug mode
                if settings.DEBUG:
                    with st.expander("Technical Details"):
                        st.code(str(e), language="text")
                        st.code(traceback.format_exc(), language="python")
    
    def display_assistant_response(self, response: str, context: Dict[str, Any]):
        """Display the assistant's response with visualizations and expandable panels"""
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display in chat
        with st.chat_message("assistant", avatar="🧞"):
            # Check if there's an error in the context
            if context.get("error"):
                st.error(response)
                
                # Show error details in debug mode
                if settings.DEBUG:
                    with st.expander("Error Details"):
                        st.code(context["error"], language="text")
            else:
                st.markdown(response)
            
            # Display results and download option if available
            if "sql_results" in context and isinstance(context["sql_results"], pd.DataFrame) and not context["sql_results"].empty:
                # # Display download link for CSV
                # st.markdown(self.generate_csv_download_link(context["sql_results"]), unsafe_allow_html=True)
                
                # Display a sample of the results
                with st.expander("📋 Preview Results"):
                    st.dataframe(context["sql_results"].head(10))
            
            # New feature: Show if this used in-memory data
            if context.get("use_in_memory_data", False):
                st.info("🧠 This answer also used data already available in our conversation.")
            
            # Display visualization if enabled and results available
            if ("sql_results" in context and 
                    isinstance(context["sql_results"], pd.DataFrame) and 
                    not context["sql_results"].empty):
                try:
                    # Create tabs for different visualizations
                    viz_tabs = st.tabs(["Table"])
                    
                    with viz_tabs[0]:
                        st.dataframe(context["sql_results"])
                            
                except Exception as e:
                    if settings.DEBUG:
                        st.error(f"Error creating visualization: {str(e)}")
        
        # Display SQL query if enabled
        if st.session_state.show_sql and "sql_query" in context and context["sql_query"]:
            with st.expander("📝 SQL Query"):
                st.code(context["sql_query"], language="sql")
        
        # Display query results in debug mode
        if settings.DEBUG and "sql_results" in context and isinstance(context["sql_results"], pd.DataFrame):
            with st.expander("🔍 Query Results (First 10 rows)"):
                st.dataframe(context["sql_results"].head(10))
        
        # New feature: Display in-memory context summary in debug mode
        if settings.DEBUG:
            with st.expander("🧠 Conversation Context"):
                context_summary = data_context_service.get_context_summary(st.session_state.conversation_id)
                
                # Show tables in memory
                st.subheader("In-Memory Tables")
                for table in context_summary.get('tables', []):
                    st.markdown(f"**{table['name']}** ({table['rows']} rows, {len(table['column_names'])} columns)")
                    st.json(table['sample'])
                
                # Show variables
                if context_summary.get('variables'):
                    st.subheader("Variables")
                    st.json(context_summary['variables'])
                
                # Show recent history
                if context_summary.get('history'):
                    st.subheader("Recent Context History")
                    for entry in context_summary['history']:
                        st.markdown(f"- **{entry['action']}**: {entry['summary']} ({entry['timestamp']})")
        
        # Display timing information if enabled
        if st.session_state.show_timing and "timing" in context and context["timing"]:
            with st.expander("⏱️ Timing Information"):
                timing_df = pd.DataFrame({
                    'Step': list(context["timing"].keys()),
                    'Time (seconds)': list(context["timing"].values())
                })
                st.dataframe(timing_df)
                
                # Timing visualization
                fig, ax = plt.subplots()
                bars = ax.barh(timing_df['Step'], timing_df['Time (seconds)'])
                ax.set_xlabel('Time (seconds)')
                ax.set_title('Processing Time by Step')
                
                # Add time values at the end of each bar
                for i, bar in enumerate(bars):
                    value = timing_df['Time (seconds)'].iloc[i]
                    ax.text(value + 0.05, bar.get_y() + bar.get_height()/2, 
                            f'{value:.2f}s', va='center')
                
                st.pyplot(fig)
    
    def run(self):
        """Run the Streamlit UI application"""
        # Setup sidebar
        self.setup_sidebar()
        
        # Main area
        st.markdown('<h1 class="main-header"> SQL Genie 🧞‍♂️</h1>', unsafe_allow_html=True)
        st.markdown('<h5 class="sub-header">Your wish for SQL queries, granted by Genie</h5>', unsafe_allow_html=True)
        
        # Show connection warnings if needed
        if st.session_state.connection_status in ["failed", "error"]:
            st.warning("⚠️ Database connection issues detected. Some features may not work correctly.")
        
        if st.session_state.llm_status == "incomplete":
            st.warning("⚠️ Azure OpenAI configuration is incomplete. Please check settings in the sidebar.")
        
        # Show data enrichment panel if enabled
        if st.session_state.data_enrichment_mode:
            self.show_data_enrichment_panel()
        
        # Display chat messages
        self.display_messages()
        
        # Input field for user query
        user_input = st.chat_input("Ask about your data... 💰")
        
        if user_input:
            self.handle_user_input(user_input)