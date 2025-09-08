import boto3
import os
from typing import Optional
from botocore.client import Config
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from COURSEGEN.data_models.document_model import Document
from COURSEGEN.data_models.file_data_model import FileDataModel

class CloudflareR2:
    def __init__(self):
        # Use environment variables for credentials and configuration
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        self.access_key_id = os.getenv("CLOUDFLARE_ACCESS_KEY_ID", "")
        self.secret_access_key = os.getenv("CLOUDFLARE_SECRET_ACCESS_KEY", "")
        self.bucket_name = os.getenv("CLOUDFLARE_BUCKET", "")
        
        if not all([self.access_key_id, self.secret_access_key, self.bucket_name]):
            raise ValueError("Cloudflare R2 credentials are not set.")

        self.s3_client = boto3.client(
            's3',
            endpoint_url=f'https://{self.account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(signature_version='s3v4')
        )

    def upload_file(
        self,
        file_path: str,
        file_name: str,
        metadata: Optional[FileDataModel] = None,
    ) -> Document:
        """Uploads a file to Cloudflare R2 and returns a Document with metadata."""
        try:
            metadata_dict = metadata.dict() if metadata else {}
            metadata_dict.setdefault("file_path", file_path)
            extra_args = {"Metadata": metadata_dict} if metadata_dict else None
            self.s3_client.upload_file(
                file_path, self.bucket_name, file_name, ExtraArgs=extra_args
            )
            # Updated to use the free Cloudflare domain pattern
            public_url = f"https://{self.bucket_name}.{self.account_id}.r2.dev/{file_name}"
            document_data = {**metadata_dict, "file_name": file_name, "cloudflare_url": public_url}
            return Document(**document_data)
        except (NoCredentialsError, PartialCredentialsError):
            print("Error: AWS credentials not found.")
            raise
        except Exception as e:
            print(f"Error uploading file to R2: {e}")
            raise

if __name__ == "__main__":
    # Example usage (requires environment variables to be set)
    # You can test this by creating a dummy file and setting up your .env file
    if os.getenv("CLOUDFLARE_ACCOUNT_ID"):
        r2_service = CloudflareR2()
        # Create a dummy file for testing
        with open("test.txt", "w") as f:
            f.write("This is a test file.")
        
        try:
            metadata = FileDataModel(
                file_path="test.txt",
                course_code="TEST101",
                department="TEST",
                level="100",
                semester="1",
                type="lecture",
            )
            doc = r2_service.upload_file("test.txt", "test.txt", metadata=metadata)
            print(f"File uploaded successfully. URL: {doc.cloudflare_url}")
        except Exception as e:
            print(f"Upload failed: {e}")
        finally:
            os.remove("test.txt")