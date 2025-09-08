import firebase_admin
from firebase_admin import credentials, firestore
from typing import Optional, List
from COURSEGEN.data_models.course_model import CourseModel
from COURSEGEN.data_models.question_model import Question
from COURSEGEN.data_models.document_model import Document
from COURSEGEN.config import load_config


class FireStore:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FireStore, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not firebase_admin._apps:
            config = load_config()
            try:
                cred_path = config.firestore_service_account
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                print("Firebase initialized successfully.")
            except Exception as e:
                print(f"Error initializing Firebase: {e}")
                self.db = None
        else:
            self.db = firestore.client()

    def get_course_by_code(self, course_code: str) -> Optional[CourseModel]:
        """Fetches a single course from Firestore by its course code."""
        if not self.db:
            return None
        try:
            doc_ref = self.db.collection("course_data").document(course_code)
            doc = doc_ref.get()
            if doc.exists:
                return CourseModel(**doc.to_dict())
            return None
        except Exception as e:
            print(f"Error fetching course by code: {e}")
            return None

    def get_all_documents(self) -> List[Document]:
        """Fetches all document metadata from the 'documents' collection."""
        if not self.db:
            return []
        try:
            docs_ref = self.db.collection("documents").order_by("upload_timestamp", direction=firestore.Query.DESCENDING)
            docs = docs_ref.stream()
            return [Document(**doc.to_dict()) for doc in docs]
        except Exception as e:
            print(f"Error fetching all documents: {e}")
            return []

    def add_document(self, document: Document):
        """Adds a new document's metadata to Firestore."""
        if not self.db:
            return
        try:
            doc_ref = self.db.collection("documents").document(document.file_name)
            doc_ref.set(document.model_dump())
            print(f"Successfully added document: {document.file_name}")
        except Exception as e:
            print(f"Error adding document: {e}")

    def set_question(self, question: Question):
        """Saves a generated question to Firestore."""
        if not self.db:
            return
        try:
            doc_ref = self.db.collection("question_data").document(question.course_code).collection(question.difficulty).document()
            doc_ref.set(question.model_dump())
        except Exception as e:
            print(f"Error setting question: {e}")

if __name__ == "__main__":
    store = FireStore()
    if store.db:
        # Example: Fetch a specific course
        course = store.get_course_by_code("EEE 301")
        if course:
            print("Found course:", course.title)
        else:
            print("Course not found.")
        
        # Example: List all documents
        all_docs = store.get_all_documents()
        print(f"Found {len(all_docs)} documents.")
