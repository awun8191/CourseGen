"""Enhanced caching service with real-time capabilities for RAG pipeline."""

import json
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict

from .cache import Cache
from ..logging_utils import get_rag_logger

@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[float] = None  # Time to live in seconds
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1

class EnhancedCache:
    """Enhanced caching service with real-time capabilities, TTL, and tagging."""
    
    def __init__(self, cache_file: str = "enhanced_cache.json", default_ttl: Optional[float] = None):
        self.cache_file = cache_file
        self.default_ttl = default_ttl
        self.base_cache = Cache(cache_file)
        self.logger = get_rag_logger("EnhancedCache")
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._load_from_disk()
    
    def _load_from_disk(self):
        """Load cache entries from disk into memory."""
        try:
            disk_data = self.base_cache.read_cache()
            for key, entry_data in disk_data.items():
                if isinstance(entry_data, dict) and 'created_at' in entry_data:
                    # This is an enhanced cache entry
                    entry = CacheEntry(**entry_data)
                    if not entry.is_expired():
                        self._memory_cache[key] = entry
                else:
                    # Legacy cache entry, convert it
                    self._memory_cache[key] = CacheEntry(
                        key=key,
                        value=entry_data,
                        created_at=time.time(),
                        accessed_at=time.time(),
                        ttl=self.default_ttl
                    )
            self.logger.info(f"Loaded {len(self._memory_cache)} entries from cache")
        except Exception as e:
            self.logger.warning(f"Could not load cache from disk: {e}")
            self._memory_cache = {}
    
    def _save_to_disk(self):
        """Save memory cache to disk."""
        try:
            disk_data = {}
            for key, entry in self._memory_cache.items():
                if not entry.is_expired():
                    disk_data[key] = asdict(entry)
            self.base_cache.write_cache(disk_data)
        except Exception as e:
            self.logger.error(f"Failed to save cache to disk: {e}")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from cache."""
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if entry.is_expired():
                self.logger.debug(f"Cache entry expired: {key}")
                del self._memory_cache[key]
                return default
            
            entry.touch()
            self.logger.cache_hit(f"Retrieved {key}")
            return entry.value
        
        self.logger.cache_miss(f"Key not found: {key}")
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, tags: List[str] = None) -> None:
        """Set a value in cache."""
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            ttl=ttl or self.default_ttl,
            tags=tags or []
        )
        
        self._memory_cache[key] = entry
        self.logger.info(f"Cached {key} (TTL: {entry.ttl}s)")
        
        # Periodically save to disk
        if len(self._memory_cache) % 10 == 0:
            self._save_to_disk()
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if key in self._memory_cache:
            del self._memory_cache[key]
            self.logger.info(f"Deleted cache key: {key}")
            return True
        return False
    
    def clear(self, tags: Optional[List[str]] = None) -> int:
        """Clear cache entries, optionally filtered by tags."""
        if tags is None:
            count = len(self._memory_cache)
            self._memory_cache.clear()
            self.logger.info(f"Cleared all {count} cache entries")
            return count
        
        # Clear entries with specific tags
        keys_to_delete = []
        for key, entry in self._memory_cache.items():
            if any(tag in entry.tags for tag in tags):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self._memory_cache[key]
        
        self.logger.info(f"Cleared {len(keys_to_delete)} cache entries with tags: {tags}")
        return len(keys_to_delete)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from cache."""
        expired_keys = []
        for key, entry in self._memory_cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._memory_cache[key]
        
        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._memory_cache)
        expired_count = sum(1 for entry in self._memory_cache.values() if entry.is_expired())
        
        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired_count,
            'expired_entries': expired_count,
            'cache_file': self.cache_file,
            'default_ttl': self.default_ttl
        }
    
    def cache_function(self, ttl: Optional[float] = None, tags: List[str] = None):
        """Decorator to cache function results."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key
                key_data = {
                    'function': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }
                cache_key = self._generate_key(key_data)
                
                # Try to get from cache
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                self.logger.processing(f"Executing {func.__name__}")
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl, tags=tags)
                
                return result
            return wrapper
        return decorator
    
    def cache_pdf_processing(self, pdf_path: str, operation: str) -> str:
        """Generate a cache key for PDF processing operations."""
        file_stat = Path(pdf_path).stat()
        key_data = {
            'pdf_path': str(Path(pdf_path).resolve()),
            'operation': operation,
            'file_size': file_stat.st_size,
            'modified_time': file_stat.st_mtime
        }
        return self._generate_key(key_data)
    
    def cache_embedding(self, text_hash: str, model: str) -> str:
        """Generate a cache key for embedding operations."""
        return self._generate_key('embedding', text_hash, model)
    
    def save(self):
        """Explicitly save cache to disk."""
        self._save_to_disk()
        self.logger.success("Cache saved to disk")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save cache."""
        self.cleanup_expired()
        self._save_to_disk()
        if exc_type is not None:
            self.logger.error(f"Error in cache context: {exc_val}")

# Global enhanced cache instance
_enhanced_cache = None

def get_enhanced_cache(cache_file: str = None, default_ttl: Optional[float] = 3600) -> EnhancedCache:
    """Get or create the global enhanced cache instance."""
    global _enhanced_cache
    if _enhanced_cache is None:
        if cache_file is None:
            # Default to gemini_cache directory
            from pathlib import Path
            cache_dir = Path(__file__).parent.parent.parent.parent / "data" / "gemini_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = str(cache_dir / "enhanced_cache.json")
        _enhanced_cache = EnhancedCache(cache_file, default_ttl)
    return _enhanced_cache

# Convenience functions
def cache_get(key: str, default: Any = None) -> Any:
    """Get a value from the global cache."""
    return get_enhanced_cache().get(key, default)

def cache_set(key: str, value: Any, ttl: Optional[float] = None, tags: List[str] = None) -> None:
    """Set a value in the global cache."""
    get_enhanced_cache().set(key, value, ttl, tags)

def cache_delete(key: str) -> bool:
    """Delete a key from the global cache."""
    return get_enhanced_cache().delete(key)

def cache_clear(tags: Optional[List[str]] = None) -> int:
    """Clear the global cache."""
    return get_enhanced_cache().clear(tags)