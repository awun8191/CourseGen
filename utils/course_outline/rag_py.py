from google import genai
from pypdf import PdfReader
import re
import os
import chromadb
from PIL import Image
import pytesseract
import io
import logging
from typing import List, Optional
import hashlib
from pathlib import Path
from datetime import datetime
import requests
import shutil  # <-- NEW: For recursively deleting directories

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('course_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CourseOutlineGenerator:
    """
    A comprehensive RAG system for generating course outlines from single or multiple PDF materials,
    supporting both local files and online URLs, with optional automatic database cleanup.
    """

    def __init__(self, api_key: Optional[str] = None, db_path: str = "./chroma_db_data"):
        """Initialize the course outline generator."""
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        self.db_path = db_path
        self.client = None
        self.generative_model = None
        self._cleanup_on_exit = False  # <-- NEW: Safety flag for cleanup
        self._setup_api()
        self._setup_database()

    ## NEW: Context manager methods for guaranteed cleanup ##
    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and perform cleanup if enabled."""
        if self._cleanup_on_exit:
            self.cleanup_database()

    def enable_cleanup_on_exit(self):
        """Explicitly enable database deletion upon exiting a 'with' block."""
        logger.warning(f"Database cleanup enabled. The directory '{self.db_path}' will be deleted on exit.")
        self._cleanup_on_exit = True

    def cleanup_database(self):
        """Deletes the entire ChromaDB persistence directory."""
        db_path = Path(self.db_path)
        if db_path.exists() and db_path.is_dir():
            logger.info(f"Cleaning up and deleting database directory: {db_path.resolve()}")
            try:
                shutil.rmtree(db_path)
                logger.info("Database directory successfully deleted.")
            except Exception as e:
                logger.error(f"Error while deleting database directory '{db_path}': {e}")
        else:
            logger.info("Database directory not found, no cleanup needed.")

    # --- All other methods from the previous version remain the same ---
    # ... ( _setup_api, _setup_database, _is_url, _generate_collection_name, etc.) ...
    def _setup_api(self):
        if not self.api_key:
            error_msg = "GOOGLE_API_KEY not found. Please set it as an environment variable."
            logger.error(error_msg)
            raise ValueError(error_msg)
        try:
            # New Google GenAI client
            self.genai_client = genai.Client(api_key=self.api_key)
            self.model_name = "gemini-2.5-flash-preview-04-17"
            logger.info("Google GenAI client configured successfully using gemini-2.5-flash-preview-04-17")
        except Exception as e:
            logger.error(f"Failed to configure Google AI: {e}")
            raise

    def _setup_database(self):
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            logger.info(f"ChromaDB initialized with persistent storage at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _is_url(self, source: str) -> bool:
        return source.lower().startswith('http://') or source.lower().startswith('https://')

    def _generate_collection_name(self, source: str) -> str:
        if self._is_url(source):
            try:
                response = requests.head(source, timeout=10)
                response.raise_for_status()
                content_length = response.headers.get('Content-Length', '0')
                etag = response.headers.get('ETag', '')
                file_info = f"{source}_{content_length}_{etag}"
                safe_stem = re.sub(r'[^a-zA-Z0-9_-]', '_', Path(source).stem)
            except requests.RequestException as e:
                logger.error(f"Could not fetch headers for URL {source}: {e}")
                file_info = source
                safe_stem = "url_source"
        else:
            file_path = Path(source)
            if not file_path.exists():
                raise FileNotFoundError(f"PDF file not found: {source}")
            file_info = f"{file_path.resolve()}_{file_path.stat().st_mtime}"
            safe_stem = re.sub(r'[^a-zA-Z0-9_-]', '_', file_path.stem)
        file_hash = hashlib.md5(file_info.encode()).hexdigest()
        return f"course_{safe_stem}_{file_hash[:12]}"

    def extract_text_from_pdf(self, source: str) -> str:
        source_name = Path(source).name if not self._is_url(source) else source
        pdf_reader = None
        try:
            if self._is_url(source):
                logger.info(f"Downloading PDF from URL: {source}")
                response = requests.get(source, timeout=30)
                response.raise_for_status()
                if 'application/pdf' not in response.headers.get('Content-Type', ''):
                    raise TypeError(f"Content at URL {source} is not a PDF.")
                pdf_file = io.BytesIO(response.content)
                pdf_reader = PdfReader(pdf_file)
            else:
                pdf_reader = PdfReader(source)
            extracted_text, total_pages = "", len(pdf_reader.pages)
            logger.info(f"Processing '{source_name}' with {total_pages} pages")
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 50:
                        extracted_text += page_text + "\n"
                    else:
                        ocr_text = self._ocr_page(page)
                        if ocr_text: extracted_text += ocr_text + "\n"
                except Exception as page_error:
                    logger.error(f"Error processing page {i + 1} of '{source_name}': {page_error}")
                    continue
            if not extracted_text.strip(): raise ValueError(f"No text could be extracted from '{source_name}'.")
            logger.info(f"Successfully extracted {len(extracted_text)} characters from '{source_name}'")
            print(extracted_text)
            return extracted_text
        except requests.RequestException as e:
            logger.error(f"Failed to download or access URL {source}: {e}")
            raise
        except FileNotFoundError:
            logger.error(f"Local PDF file not found: {source}")
            raise
        except Exception as e:
            logger.error(f"Error during PDF processing for '{source_name}': {e}")
            raise

    def _ocr_page(self, page) -> str:
        try:
            if hasattr(page, 'images') and page.images:
                page_ocr_text = ""
                for img_index, img in enumerate(page.images):
                    try:
                        image = Image.open(io.BytesIO(img.data))
                        image = image.convert('L')
                        ocr_text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
                        if ocr_text.strip(): page_ocr_text += ocr_text + "\n"
                    except Exception as img_error:
                        logger.debug(f"Failed to process image {img_index} on page: {img_error}")
                        continue
                return page_ocr_text
            return ""
        except Exception as e:
            logger.warning(f"OCR process failed for page: {e}")
            return ""

    def intelligent_chunking(self, text: str, max_chunk_size_words: int = 300, overlap_words: int = 50) -> List[str]:
        if not text.strip(): return []
        cleaned = re.sub(r"\n?-+ PAGE BREAK -+\n?", "\n\n", text.strip())
        paragraphs = re.split(r'\n\s*\n', cleaned)
        chunks, current_chunk = [], ""
        for para in paragraphs:
            if not para.strip(): continue
            if len(current_chunk.split()) + len(para.split()) <= max_chunk_size_words:
                current_chunk += "\n\n" + para
            elif current_chunk:
                chunks.append(current_chunk.strip())
                overlap = " ".join(current_chunk.split()[-overlap_words:])
                current_chunk = overlap + "\n\n" + para
            else:
                current_chunk = para
        if current_chunk: chunks.append(current_chunk.strip())
        chunks = [chunk for chunk in chunks if len(chunk.split()) >= 20]
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks

    def embed_text(self, text_list: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> Optional[List]:
        if not text_list: return None
        model_name, max_retries = "text-embedding-004", 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating embeddings for {len(text_list)} chunks (attempt {attempt + 1}/{max_retries})")
                resp = self.genai_client.models.embed_content(
                    model=model_name,
                    contents=text_list,
                    config={"task_type": task_type.lower()},
                )
                embeddings = [e.values for e in (resp.embeddings or [])]
                logger.info(f"Generated {len(embeddings)} embeddings")
                return embeddings
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1: logger.error("All embedding attempts failed")
                return None
        return None

    def process_document(self, source: str, force_reprocess: bool = False) -> bool:
        source_name = Path(source).name if not self._is_url(source) else source
        try:
            collection_name = self._generate_collection_name(source)
            try:
                collection = self.client.get_collection(name=collection_name)
                if force_reprocess:
                    logger.info(f"Force reprocessing enabled. Deleting collection for '{source_name}'.")
                    self.client.delete_collection(name=collection_name)
                elif collection.count() > 0:
                    logger.info(f"Using existing collection for '{source_name}'.")
                    return True
            except Exception:
                logger.info(f"Collection for '{source_name}' not found. A new one will be created.")
                pass
            logger.info(f"Processing new document: {source_name}...")
            document_text = self.extract_text_from_pdf(source)
            if not (text_chunks := self.intelligent_chunking(document_text)): raise ValueError(
                "No valid chunks created.")
            if not (chunk_embeddings := self.embed_text(text_chunks, "RETRIEVAL_DOCUMENT")): raise ValueError(
                "Failed to generate embeddings.")
            collection = self.client.get_or_create_collection(name=collection_name)
            chunk_ids = [f"chunk_{i}" for i in range(len(text_chunks))]
            metadata = [{"source_file": source_name, "chunk_index": i} for i in range(len(text_chunks))]
            collection.add(embeddings=chunk_embeddings, documents=text_chunks, ids=chunk_ids, metadatas=metadata)
            logger.info(f"Successfully processed '{source_name}' into collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error processing document '{source_name}': {e}")
            return False

    def process_multiple_documents(self, sources: List[str], force_reprocess: bool = False) -> bool:
        logger.info(f"Starting processing for {len(sources)} documents.")
        all_successful = True
        for source in sources:
            if not self.process_document(source, force_reprocess):
                logger.error(f"Failed to process {source}. It will be skipped.")
                all_successful = False
        if all_successful:
            logger.info("All documents were processed successfully.")
        else:
            logger.warning("Some documents failed to process.")
        return all_successful

    def retrieve_relevant_chunks(self, query: str, sources: List[str], top_k: int = 15) -> List[str]:
        all_results = []
        logger.info(f"Retrieving chunks from {len(sources)} source(s) for query: '{query[:50]}...'")
        if not (query_embedding := self.embed_text([query], "RETRIEVAL_QUERY")): raise ValueError(
            "Failed to generate query embedding.")
        for source in sources:
            source_name = Path(source).name if not self._is_url(source) else source
            try:
                collection_name = self._generate_collection_name(source)
                collection = self.client.get_collection(name=collection_name)
                if collection.count() == 0:
                    logger.warning(f"Collection for '{source_name}' is empty. Skipping.")
                    continue
                n_results = min(top_k, collection.count())
                results = collection.query(query_embeddings=query_embedding, n_results=n_results,
                                           include=['documents', 'distances'])
                for doc, dist in zip(results['documents'][0], results['distances'][0]):
                    all_results.append({'document': doc, 'distance': dist})
                logger.info(f"Retrieved {len(results['documents'][0])} chunks from '{source_name}'.")
            except Exception as e:
                logger.warning(f"Could not retrieve from '{source_name}': {e}.")
                continue
        all_results.sort(key=lambda x: x['distance'])
        final_chunks = [res['document'] for res in all_results[:top_k]]
        logger.info(f"Combined and sorted all chunks. Returning the top {len(final_chunks)} results.")
        return final_chunks

    def generate_course_outline(self, sources: List[str], custom_prompt: Optional[str] = None) -> str:
        try:
            logger.info(f"Starting course outline generation from {len(sources)} document(s).")
            if not isinstance(sources, list): sources = [sources]
            query = "comprehensive summary of all major topics, chapters, and key concepts for a university course syllabus"
            if not (relevant_chunks := self.retrieve_relevant_chunks(query, sources, top_k=20)):
                raise ValueError("No content could be retrieved from the document(s).")
            context = "\n\n---\n\n".join(relevant_chunks)
            logger.info(f"Using {len(relevant_chunks)} chunks for context ({len(context)} characters).")
            default_prompt = f"""
## ROLE ##
You are an expert curriculum designer and university professor tasked with creating a course syllabus.

## INSTRUCTIONS ##
1.  **Analyze** the provided SOURCE MATERIAL, which contains excerpts from multiple documents.
2.  **Synthesize** the information to create a single, unified, and logically structured course outline.
3.  **Merge** related topics from different sources. Do not treat them as separate subjects.
4.  **Generate** a suitable Course Title and comprehensive Course Description that encompasses all source material.
5.  **Infer** the main learning objectives for the course in a clear and concise manner.
6.  **Format** the entire output in clean, professional Markdown.

## SOURCE MATERIAL ##
---
{context}
---

## TASK ##
Based on the instructions and the source material provided above, generate the complete course outline now. Start directly with the course title.

**Generated Course Outline (Markdown Format):**
"""
            prompt = custom_prompt or default_prompt
            logger.info("Generating synthesized course outline with the Gemini model...")
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            # response.text contains the combined text output
            class _R: pass
            _r = _R()
            _r.text = getattr(response, 'text', None) or ''
            response = _r
            if not response.text: raise ValueError("Received an empty response from the AI model.")
            logger.info("Course outline generated successfully.")
            return response.text
        except Exception as e:
            logger.error(f"Error during course outline generation: {e}")
            raise

    def save_outline_to_file(self, outline: str, sources: List[str]) -> str:
        try:
            first_source = sources[0]
            base_name = Path(first_source).stem if not self._is_url(first_source) else "online_course"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"course_outline_{base_name}_{timestamp}.md"
            output_path = Path(output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Course Outline\n\n**Source Documents:**\n")
                for source in sources:
                    source_name = Path(source).name if not self._is_url(source) else source
                    f.write(f"- `{source_name}`\n")
                f.write(f"\n**Generated On:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n")
                f.write(outline)
            logger.info(f"Outline saved to: {output_path.resolve()}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save outline to file: {e}")
            raise


## MODIFIED ##
def main():
    """Main function to demonstrate the CourseOutlineGenerator with cleanup."""
    # Use a 'with' block to ensure resources are managed and cleaned up correctly.
    try:
        with CourseOutlineGenerator(api_key="AIzaSyB1qxAZA6G327lxiaI8pwkFKYRe1JDRz0o") as generator:
            # --- CONFIGURATION ---
            from COURSEGEN.config import load_config
            config = load_config()
            document_sources = [
                config.document_source,
            ]
            force_reprocessing = False

            # --- NEW: Cleanup Configuration ---
            # Set to True to delete the database folder after the script finishes.
            # Set to False to keep the database for caching on future runs.
            cleanup_database_on_exit = True

            if cleanup_database_on_exit:
                generator.enable_cleanup_on_exit()
            # ---------------------

            # Pre-flight check for sources
            valid_sources = [
                s for s in document_sources if generator._is_url(s) or Path(s).exists()
            ]
            if len(valid_sources) != len(document_sources):
                logger.warning("Some sources were not found and will be skipped.")
            if not valid_sources:
                logger.error("No valid input files or URLs found. Aborting.")
                return

            # 1. Process all documents
            generator.process_multiple_documents(valid_sources, force_reprocess=force_reprocessing)

            # 2. Generate the synthesized course outline
            logger.info("Generating synthesized course outline from all sources...")
            outline = generator.generate_course_outline(valid_sources)

            # 3. Display and save the result
            print("\n" + "=" * 80)
            print("GENERATED COURSE OUTLINE")
            print("=" * 80)
            print(outline)
            print("=" * 80)

            output_file = generator.save_outline_to_file(outline, valid_sources)
            print(f"\n✅ Outline successfully saved to: {output_file}")

    except ValueError as ve:
        logger.error(f"Configuration or Processing Error: {ve}")
    except Exception as e:
        logger.error(f"An unexpected application error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
    # The 'with' block automatically handles the cleanup when the block is exited.


if __name__ == "__main__":
    main()
