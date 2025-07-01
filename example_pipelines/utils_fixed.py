from pathlib import Path

def get_project_root():
    """
    Get the project root directory reliably.
    
    This function traverses up from the current file's location
    until it finds a directory containing the 'datasets' folder,
    which indicates the project root.
    
    Returns:
        Path: The project root directory
        
    Raises:
        FileNotFoundError: If project root cannot be determined
    """
    # Start from the directory containing this utils.py file
    current_path = Path(__file__).resolve().parent
    
    # Traverse up the directory tree
    for parent in [current_path] + list(current_path.parents):
        if (parent / "datasets").exists():
            return parent
    
    # Fallback: if no datasets folder found, raise an error
    raise FileNotFoundError(
        "Could not find project root. Please ensure 'datasets' folder exists in the project."
    )