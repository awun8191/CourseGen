"""Progress tracking and resume functionality for RAG pipeline."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum

class ProcessingStatus(Enum):
    """Status of file processing."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class FileProgress:
    """Progress information for a single file."""
    file_path: str
    status: ProcessingStatus
    chunks_extracted: int = 0
    embeddings_generated: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    last_updated: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()

@dataclass
class PipelineProgress:
    """Overall pipeline progress information."""
    session_id: str
    start_time: str
    last_updated: str
    total_files: int
    completed_files: int
    failed_files: int
    skipped_files: int
    total_chunks: int
    total_embeddings: int
    processing_parameters: Dict[str, Any]
    file_progress: Dict[str, FileProgress]
    
    def __post_init__(self):
        # Convert FileProgress objects from dict if needed
        if self.file_progress and isinstance(list(self.file_progress.values())[0], dict):
            self.file_progress = {
                k: FileProgress(**v) if isinstance(v, dict) else v 
                for k, v in self.file_progress.items()
            }

class ProgressTracker:
    """Tracks and manages processing progress with resume capability."""
    
    def __init__(self, progress_file: str = None, session_id: str = None):
        if progress_file is None:
            progress_dir = Path(__file__).parent.parent.parent / "data" / "progress"
            progress_dir.mkdir(parents=True, exist_ok=True)
            progress_file = str(progress_dir / "pipeline_progress.json")
        
        self.progress_file = Path(progress_file)
        self.session_id = session_id or f"session_{int(time.time())}"
        self.progress: Optional[PipelineProgress] = None
        self._lock_file = self.progress_file.with_suffix('.lock')
        
    def initialize_session(self, total_files: int, processing_params: Dict[str, Any]) -> str:
        """Initialize a new processing session."""
        self.progress = PipelineProgress(
            session_id=self.session_id,
            start_time=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            total_files=total_files,
            completed_files=0,
            failed_files=0,
            skipped_files=0,
            total_chunks=0,
            total_embeddings=0,
            processing_parameters=processing_params,
            file_progress={}
        )
        self.save_progress()
        return self.session_id
    
    def load_progress(self) -> Optional[PipelineProgress]:
        """Load existing progress from file."""
        if not self.progress_file.exists():
            return None
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert status strings back to enums
            if 'file_progress' in data:
                for file_path, file_data in data['file_progress'].items():
                    if 'status' in file_data:
                        file_data['status'] = ProcessingStatus(file_data['status'])
            
            self.progress = PipelineProgress(**data)
            return self.progress
        except Exception as e:
            print(f"Warning: Could not load progress file: {e}")
            return None
    
    def save_progress(self):
        """Save current progress to file."""
        if not self.progress:
            return
        
        # Create a lock file to prevent concurrent writes
        try:
            self._lock_file.touch()
            
            # Convert to dict and handle enums
            data = asdict(self.progress)
            
            # Convert enum values to strings
            for file_path, file_data in data['file_progress'].items():
                if 'status' in file_data:
                    file_data['status'] = file_data['status'].value
            
            data['last_updated'] = datetime.now().isoformat()
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.progress_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            temp_file.replace(self.progress_file)
            
        finally:
            # Remove lock file
            if self._lock_file.exists():
                self._lock_file.unlink()
    
    def update_file_status(self, file_path: str, status: ProcessingStatus, 
                          chunks: int = 0, embeddings: int = 0, 
                          processing_time: float = 0.0, error: str = None,
                          metadata: Dict[str, Any] = None):
        """Update the status of a specific file."""
        if not self.progress:
            return
        
        file_key = str(Path(file_path).resolve())
        
        if file_key not in self.progress.file_progress:
            self.progress.file_progress[file_key] = FileProgress(
                file_path=file_path,
                status=status,
                metadata=metadata or {}
            )
        
        file_progress = self.progress.file_progress[file_key]
        old_status = file_progress.status
        
        # Update file progress
        file_progress.status = status
        file_progress.chunks_extracted = chunks
        file_progress.embeddings_generated = embeddings
        file_progress.processing_time = processing_time
        file_progress.error_message = error
        file_progress.last_updated = datetime.now().isoformat()
        
        if metadata:
            file_progress.metadata.update(metadata)
        
        # Update overall counters
        if old_status != status:
            if old_status == ProcessingStatus.COMPLETED:
                self.progress.completed_files -= 1
            elif old_status == ProcessingStatus.FAILED:
                self.progress.failed_files -= 1
            elif old_status == ProcessingStatus.SKIPPED:
                self.progress.skipped_files -= 1
            
            if status == ProcessingStatus.COMPLETED:
                self.progress.completed_files += 1
                self.progress.total_chunks += chunks
                self.progress.total_embeddings += embeddings
            elif status == ProcessingStatus.FAILED:
                self.progress.failed_files += 1
            elif status == ProcessingStatus.SKIPPED:
                self.progress.skipped_files += 1
        
        self.save_progress()
    
    def get_pending_files(self, all_files: List[str]) -> List[str]:
        """Get list of files that still need processing."""
        if not self.progress:
            return all_files
        
        pending_files = []
        for file_path in all_files:
            file_key = str(Path(file_path).resolve())
            if file_key not in self.progress.file_progress:
                pending_files.append(file_path)
            else:
                status = self.progress.file_progress[file_key].status
                if status in [ProcessingStatus.PENDING, ProcessingStatus.FAILED]:
                    pending_files.append(file_path)
        
        return pending_files
    
    def get_completed_files(self) -> List[str]:
        """Get list of successfully completed files."""
        if not self.progress:
            return []
        
        return [
            fp.file_path for fp in self.progress.file_progress.values()
            if fp.status == ProcessingStatus.COMPLETED
        ]
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of current progress."""
        if not self.progress:
            return {}
        
        return {
            'session_id': self.progress.session_id,
            'total_files': self.progress.total_files,
            'completed_files': self.progress.completed_files,
            'failed_files': self.progress.failed_files,
            'skipped_files': self.progress.skipped_files,
            'pending_files': self.progress.total_files - self.progress.completed_files - self.progress.failed_files - self.progress.skipped_files,
            'total_chunks': self.progress.total_chunks,
            'total_embeddings': self.progress.total_embeddings,
            'completion_percentage': (self.progress.completed_files / self.progress.total_files * 100) if self.progress.total_files > 0 else 0,
            'start_time': self.progress.start_time,
            'last_updated': self.progress.last_updated
        }
    
    def can_resume(self) -> bool:
        """Check if there's a session that can be resumed."""
        progress = self.load_progress()
        if not progress:
            return False
        
        # Check if there are pending or failed files
        pending_count = progress.total_files - progress.completed_files - progress.skipped_files
        return pending_count > 0
    
    def cleanup_session(self):
        """Clean up the current session (remove progress file)."""
        if self.progress_file.exists():
            self.progress_file.unlink()
        if self._lock_file.exists():
            self._lock_file.unlink()
    
    def export_results(self, export_path: str = None) -> str:
        """Export processing results to a summary file."""
        if not self.progress:
            return ""
        
        if export_path is None:
            export_path = str(self.progress_file.parent / f"results_{self.session_id}.json")
        
        summary = self.get_progress_summary()
        summary['detailed_results'] = {
            file_key: asdict(file_progress) 
            for file_key, file_progress in self.progress.file_progress.items()
        }
        
        # Convert enum values to strings for JSON serialization
        for file_data in summary['detailed_results'].values():
            if 'status' in file_data:
                file_data['status'] = file_data['status'].value
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return export_path

# Convenience functions
def create_progress_tracker(progress_file: str = None, session_id: str = None) -> ProgressTracker:
    """Create a new progress tracker instance."""
    return ProgressTracker(progress_file, session_id)

def resume_or_create_session(progress_file: str = None) -> ProgressTracker:
    """Resume existing session or create new one."""
    tracker = ProgressTracker(progress_file)
    existing_progress = tracker.load_progress()
    
    if existing_progress and tracker.can_resume():
        print(f"Resuming session {existing_progress.session_id}")
        print(f"Progress: {existing_progress.completed_files}/{existing_progress.total_files} files completed")
        return tracker
    else:
        print("Starting new session")
        return tracker