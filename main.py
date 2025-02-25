import os
import logging
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import app UI (after environment variables are loaded)
from ui.app_ui import StreamlitUI


def main():
    """Main entry point for the application"""
    try:
        # Initialize and run the Streamlit UI
        ui = StreamlitUI()
        ui.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()