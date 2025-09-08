"""Beautiful and descriptive logging utilities for the RAG pipeline."""

import logging
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Standard colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors and emojis to log messages."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.BRIGHT_BLACK,
        logging.INFO: Colors.BRIGHT_BLUE,
        logging.WARNING: Colors.BRIGHT_YELLOW,
        logging.ERROR: Colors.BRIGHT_RED,
        logging.CRITICAL: Colors.BG_RED + Colors.BRIGHT_WHITE,
    }
    
    LEVEL_EMOJIS = {
        logging.DEBUG: '[DEBUG]',
        logging.INFO: '[INFO] ',
        logging.WARNING: '[WARN] ',
        logging.ERROR: '[ERROR]',
        logging.CRITICAL: '[CRIT]',
    }
    
    def format(self, record):
        # Get color and emoji for the log level
        color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
        emoji = self.LEVEL_EMOJIS.get(record.levelno, '[LOG]')
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Create the formatted message - avoid Unicode characters for Windows compatibility
        try:
            formatted_msg = (
                f"{Colors.DIM}[{timestamp}]{Colors.RESET} "
                f"{emoji} {color}{record.levelname:<8}{Colors.RESET} "
                f"{Colors.CYAN}{record.name}{Colors.RESET} - "
                f"{record.getMessage()}"
            )
        except UnicodeEncodeError:
            # Fallback without colors for problematic terminals
            formatted_msg = f"[{timestamp}] {emoji} {record.levelname:<8} {record.name} - {record.getMessage()}"
        
        return formatted_msg

class RAGLogger:
    """Enhanced logger for RAG pipeline with beautiful formatting and progress tracking."""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler (plain text)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def step(self, message: str, step_num: Optional[int] = None, total_steps: Optional[int] = None):
        """Log a processing step with beautiful formatting."""
        if step_num and total_steps:
            progress = f"[{step_num}/{total_steps}]"
            self.logger.info(f"STEP {Colors.BOLD}{progress}{Colors.RESET} {message}")
        else:
            self.logger.info(f"STEP {message}")
    
    def success(self, message: str):
        """Log a success message."""
        self.logger.info(f"SUCCESS {Colors.GREEN}{message}{Colors.RESET}")
    
    def processing(self, message: str, item: str = ""):
        """Log a processing message."""
        if item:
            self.logger.info(f"PROC {message}: {Colors.YELLOW}{item}{Colors.RESET}")
        else:
            self.logger.info(f"PROC {message}")
    
    def cache_hit(self, message: str):
        """Log a cache hit."""
        self.logger.info(f"CACHE {Colors.GREEN}Hit:{Colors.RESET} {message}")
    
    def cache_miss(self, message: str):
        """Log a cache miss."""
        self.logger.info(f"CACHE {Colors.YELLOW}Miss:{Colors.RESET} {message}")
    
    def embedding(self, message: str, count: Optional[int] = None):
        """Log embedding generation."""
        if count:
            self.logger.info(f"EMBED {message} ({Colors.CYAN}{count} embeddings{Colors.RESET})")
        else:
            self.logger.info(f"EMBED {message}")
    
    def database(self, message: str):
        """Log database operations."""
        self.logger.info(f"DB {message}")
    
    def file_operation(self, operation: str, file_path: str):
        """Log file operations."""
        file_name = Path(file_path).name
        self.logger.info(f"FILE {operation}: {Colors.CYAN}{file_name}{Colors.RESET}")
    
    def progress_bar(self, current: int, total: int, message: str = ""):
        """Log progress with a visual progress bar."""
        percentage = (current / total) * 100
        bar_length = 20
        filled_length = int(bar_length * current // total)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        
        progress_msg = f"PROGRESS {Colors.CYAN}[{bar}]{Colors.RESET} {percentage:.1f}% ({current}/{total})"
        if message:
            progress_msg += f" - {message}"
        
        self.logger.info(progress_msg)
    
    def section_header(self, title: str):
        """Log a section header with decorative formatting."""
        border = "=" * (len(title) + 4)
        self.logger.info(f"\n{Colors.BRIGHT_CYAN}{border}{Colors.RESET}")
        self.logger.info(f"{Colors.BRIGHT_CYAN}| {Colors.BOLD}{title}{Colors.RESET}{Colors.BRIGHT_CYAN} |{Colors.RESET}")
        self.logger.info(f"{Colors.BRIGHT_CYAN}{border}{Colors.RESET}\n")
    
    def warning(self, message: str, *args, **kwargs):
        """Log a warning message (supports printf-style formatting)."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log an error message (supports printf-style formatting)."""
        self.logger.error(message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Log a debug message (supports printf-style formatting)."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log an info message (supports printf-style formatting)."""
        self.logger.info(message, *args, **kwargs)

# Global logger instance
_rag_logger = None

def get_rag_logger(name: str = "RAG", log_file: Optional[str] = None) -> RAGLogger:
    """Get or create the global RAG logger instance."""
    global _rag_logger
    if _rag_logger is None:
        _rag_logger = RAGLogger(name, log_file)
    return _rag_logger

# Convenience functions
def log_step(message: str, step_num: Optional[int] = None, total_steps: Optional[int] = None):
    """Log a processing step."""
    get_rag_logger().step(message, step_num, total_steps)

def log_success(message: str):
    """Log a success message."""
    get_rag_logger().success(message)

def log_processing(message: str, item: str = ""):
    """Log a processing message."""
    get_rag_logger().processing(message, item)

def log_cache_hit(message: str):
    """Log a cache hit."""
    get_rag_logger().cache_hit(message)

def log_cache_miss(message: str):
    """Log a cache miss."""
    get_rag_logger().cache_miss(message)

def log_embedding(message: str, count: Optional[int] = None):
    """Log embedding generation."""
    get_rag_logger().embedding(message, count)

def log_database(message: str):
    """Log database operations."""
    get_rag_logger().database(message)

def log_file_operation(operation: str, file_path: str):
    """Log file operations."""
    get_rag_logger().file_operation(operation, file_path)

def log_progress(current: int, total: int, message: str = ""):
    """Log progress with a visual progress bar."""
    get_rag_logger().progress_bar(current, total, message)

def log_section_header(title: str):
    """Log a section header."""
    get_rag_logger().section_header(title)