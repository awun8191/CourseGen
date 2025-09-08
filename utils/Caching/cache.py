import json
import logging
import os
from datetime import datetime


class Cache:
    """A simple JSON file based cache utility."""

    def __init__(self, cache_file: str = "Text-Cache.json"):
        """Create a new cache manager.

        Parameters
        ----------
        cache_file:
            Path to the JSON file used for storing cached data.
        """

        self.cache_file_name = cache_file
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        """Enable usage as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Handle context manager exit, logging any exception."""
        if exc_type is not None:
            self.logger.error(f"Error occurred: {exc_val}")
        return False

    def write_cache(self, data):
        """Write data to cache file."""
        try:
            with open(self.cache_file_name, 'w') as file:
                json.dump(data, file, indent=4)
            self.logger.info(f"Successfully wrote to cache: {self.cache_file_name}")
        except Exception as e:
            self.logger.error(f"Error writing to cache: {str(e)}")
            raise

    def delete_cache(self):
        """Delete the cache file if it exists."""
        try:
            if os.path.exists(self.cache_file_name):
                os.remove(self.cache_file_name)
                self.logger.info(f"Cache file deleted: {self.cache_file_name}")
            else:
                self.logger.warning(f"Cache file not found: {self.cache_file_name}")
        except Exception as e:
            self.logger.error(f"Error deleting cache: {str(e)}")
            raise

    def read_cache(self):
        """Read and return data from cache file."""
        try:
            with open(self.cache_file_name, 'r') as file:
                data = json.load(file)
            self.logger.info(f"Successfully read from cache: {self.cache_file_name}")
            return data
        except FileNotFoundError:
            self.logger.warning(f"Cache file not found: {self.cache_file_name}")
            return {}
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON format in {self.cache_file_name}")
            return {}
        except Exception as e:
            self.logger.error(f"Error reading cache: {str(e)}")
            return {}

    def refresh_daily_cache(self):
        """Deletes the cache file if it's from a previous day."""
        try:
            if os.path.exists(self.cache_file_name):
                file_date = datetime.fromtimestamp(os.path.getmtime(self.cache_file_name)).date()
                if file_date < datetime.now().date():
                    self.delete_cache()
                    self.logger.info("Cache has been refreshed for the new day.")
        except Exception as e:
            self.logger.error(f"Error refreshing daily cache: {str(e)}")
            raise

    def update_cache(self, key, value):
        """
        Update specific key-value pair in the cache while preserving other existing data.
        Creates the cache file if it doesn't exist.

        Args:
            key: The key to update or add
            value: The new value for the key
        """
        try:
            if not os.path.exists(self.cache_file_name):
                self.logger.info(f"Cache file does not exist. Creating new cache file: {self.cache_file_name}")
                current_data = {}
            else:
                current_data = self.read_cache()
                self.logger.info(f"Updating existing cache with {len(current_data)} entries")

            current_data[key] = value
            self.write_cache(current_data)
            self.logger.info(f"Successfully updated cache key: {key}. Total entries: {len(current_data)}")
        except Exception as e:
            self.logger.error(f"Error updating cache: {str(e)}")
            raise


