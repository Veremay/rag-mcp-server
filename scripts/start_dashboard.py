import os
import sys
from pathlib import Path

from streamlit.web import cli as stcli

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.settings import load_settings


def main() -> None:
    try:
        settings = load_settings()
        port = str(settings.observability.dashboard_port)
        os.environ.setdefault("STREAMLIT_SERVER_PORT", port)
    except Exception as e:
        print(f"Warning: Failed to load settings: {e}")
        os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")

    app_path = project_root / "src" / "observability" / "dashboard" / "app.py"

    print(f"Starting dashboard at port {os.environ['STREAMLIT_SERVER_PORT']}...")

    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
