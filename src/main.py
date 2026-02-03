import sys
import os

# Add project root to sys.path to ensure absolute imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.settings import load_settings

def main():
    """
    Entry point for the Modular RAG MCP Server.
    """
    print("Modular RAG MCP Server Initializing...")
    
    try:
        settings = load_settings()
        print(f"✅ Configuration loaded successfully (LLM Provider: {settings.llm.provider})")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
        
    print("Modular RAG MCP Server Started")
    return 0

if __name__ == "__main__":
    sys.exit(main())
