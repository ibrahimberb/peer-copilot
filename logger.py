# logger.py
import streamlit as st
import sys

class StreamlitLogger:
    """Unified logging that works in both local and deployed Streamlit apps"""
    
    def __init__(self):
        self.messages = []
    
    def log(self, message, level="info"):
        """Log to both console and Streamlit UI"""
        # Always print to console
        print(message)
        
        # Also show in Streamlit UI if running in Streamlit
        if "streamlit" in sys.modules:
            if level == "info":
                st.info(message)
            elif level == "error":
                st.error(message)
            elif level == "warning":
                st.warning(message)
            elif level == "success":
                st.success(message)
    
    def info(self, message):
        self.log(message, "info")
    
    def error(self, message):
        self.log(message, "error")
    
    def warning(self, message):
        self.log(message, "warning")
    
    def success(self, message):
        self.log(message, "success")

# Create global logger instance
logger = StreamlitLogger()