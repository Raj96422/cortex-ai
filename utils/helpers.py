import hashlib
import datetime
from pathlib import Path
from typing import Union, List, Dict, Any
from utils.constants import ALLOWED_EXTENSIONS

def calculate_file_hash(file_bytes: bytes) -> str:
    """
    Calculates the SHA-256 hash of a file's binary content.
    This helps in deduplicating uploaded documents to avoid reprocessing
    identical files.
    
    Args:
        file_bytes (bytes): Binary content of the file.
        
    Returns:
        str: Hexadecimal SHA-256 hash.
    """
    sha256_hash = hashlib.sha256()
    # Update hash in 64kb chunks
    chunk_size = 65536
    for i in range(0, len(file_bytes), chunk_size):
        sha256_hash.update(file_bytes[i:i + chunk_size])
    return sha256_hash.hexdigest()

def validate_file_extension(filename: str) -> bool:
    """
    Checks if the uploaded file has a valid PDF extension.
    
    Args:
        filename (str): Name of the file.
        
    Returns:
        bool: True if the file has an allowed extension, False otherwise.
    """
    file_path = Path(filename)
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS

def format_file_size(size_in_bytes: int) -> str:
    """
    Converts a file size in bytes to a human-readable string representation.
    
    Args:
        size_in_bytes (int): Size in bytes.
        
    Returns:
        str: Human-readable size string (e.g., '12.4 MB', '450 KB').
    """
    for unit in ['Bytes', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"

def generate_markdown_transcript(chat_history: List[Dict[str, Any]]) -> str:
    """
    Converts Streamlit chat history logs into a clean, formatted Markdown document
    for user download.
    
    Args:
        chat_history (List[Dict[str, Any]]): List of chat message dictionaries.
            Each dictionary should contain 'role' and 'content' keys.
            
    Returns:
        str: Markdown formatted string of the transcript.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_lines = [
        f"# Cortex AI Chat Transcript",
        f"**Generated:** {timestamp}",
        "---",
        ""
    ]
    
    for message in chat_history:
        role = message.get("role", "").capitalize()
        content = message.get("content", "")
        # Add visual separation for user vs assistant
        md_lines.append(f"### 👤 {role}" if role == "User" else f"### 🧠 {role}")
        md_lines.append(content)
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
    return "\n".join(md_lines)
