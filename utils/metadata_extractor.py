# src/utils/metadata_extractor.py
from __future__ import annotations

import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import fitz  # PyMuPDF

@dataclass
class DocumentMetadata:
    # File info
    file_path: str
    file_name: str
    file_size: int
    file_hash: str
    created_date: str
    modified_date: str

    # PDF-specific (trimmed per request; keep only title & timestamps)
    pdf_title: Optional[str] = None
    pdf_creation_date: Optional[str] = None
    pdf_modification_date: Optional[str] = None

    # Content analysis
    page_count: int = 0
    total_characters: int = 0
    word_count: int = 0
    image_ratio: float = 0.0
    text_density: float = 0.0
    language: Optional[str] = None

    # Academic metadata
    department: Optional[str] = None
    course_code: Optional[str] = None
    course_number: Optional[str] = None
    level: Optional[str] = None
    semester: Optional[str] = None
    academic_year: Optional[str] = None

    # Classification
    document_type: Optional[str] = None
    topics: List[str] = None
    keywords: List[str] = None

    # Processing
    processing_method: str = "unknown"
    ocr_confidence: Optional[float] = None
    extraction_timestamp: str = ""

    # Quality
    readability_score: Optional[float] = None
    completeness_score: float = 0.0

    # Tags
    tags: Dict[str, Any] = None

    def __post_init__(self):
        if self.topics is None:
            self.topics = []
        if self.keywords is None:
            self.keywords = []
        if self.tags is None:
            self.tags = {}
        if not self.extraction_timestamp:
            self.extraction_timestamp = datetime.now().isoformat()

class MetadataExtractor:
    def __init__(self):
        self.course_pattern = re.compile(r"([A-Za-z]+(?:'[A-Za-z]*)?[A-Za-z]*)\s*[-_ ]*\s*(\d{2,3})", re.IGNORECASE)
        self.level_normalize = {
            r'(\d{3})L': r'\1',
            r'(\d{3})\s+LEVEL': r'\1',
            r'(\d{3})\s+level': r'\1',
        }
        self.document_type_patterns = {
            r'\b(lecture|notes?)\b': 'lecture',
            r'\b(assignment|homework|hw)\b': 'assignment',
            r'\b(exam|test|quiz|midterm|final)\b': 'exam',
            r'\b(textbook|book|manual)\b': 'textbook',
            r'\b(syllabus|outline)\b': 'syllabus',
            r'\b(lab|laboratory|practical)\b': 'lab',
            r'\b(project|report)\b': 'project',
            r'\bpq\b': 'exam',
        }

    def extract_metadata(self, file_path: str, content: str = None, pdf_analysis: Dict = None) -> DocumentMetadata:
        path = Path(file_path)
        try:
            st = path.stat()
            file_size = st.st_size
            created_date = datetime.fromtimestamp(st.st_ctime).isoformat()
            modified_date = datetime.fromtimestamp(st.st_mtime).isoformat()
        except Exception:
            file_size = 0
            created_date = ""
            modified_date = ""

        file_hash = self._hash_file(file_path)
        dm = DocumentMetadata(
            file_path=str(path.resolve()),
            file_name=path.name,
            file_size=file_size,
            file_hash=file_hash,
            created_date=created_date,
            modified_date=modified_date,
        )

        if path.suffix.lower() == ".pdf":
            self._extract_pdf_metadata(file_path, dm)

        self._extract_path_metadata(file_path, dm)
        if content:
            self._extract_content_metadata(content, dm)
        if pdf_analysis:
            self._integrate_pdf_analysis(pdf_analysis, dm)

        dm.tags = dm.tags or {}
        dm.tags['group_key'] = self._group_key(dm)
        self._calculate_quality(dm)
        return dm

    def _hash_file(self, file_path: str) -> str:
        h = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]
        except Exception:
            return "unknown"

    def _extract_pdf_metadata(self, file_path: str, dm: DocumentMetadata):
        try:
            doc = fitz.open(file_path)
            meta = doc.metadata
            dm.pdf_title = (meta.get('title') or '').strip() or None
            dm.pdf_creation_date = (meta.get('creationDate') or '').strip() or None
            dm.pdf_modification_date = (meta.get('modDate') or '').strip() or None
            dm.page_count = len(doc)
            doc.close()
        except Exception:
            pass

    def _extract_path_metadata(self, file_path: str, dm: DocumentMetadata):
        parts = Path(file_path).parts
        filename = parts[-1] if len(parts) >= 1 else ""
        stem = Path(filename).stem
        course_folder = parts[-2] if len(parts) >= 2 else ""
        semester = parts[-3] if len(parts) >= 3 else ""
        level = parts[-4] if len(parts) >= 4 else ""
        department = parts[-5] if len(parts) >= 5 else ""

        if department and re.match(r'^[A-Z]{2,4}$', department.upper()):
            dm.department = department.upper()
            dm.tags['explicit_department'] = True

        if level:
            for pattern, repl in self.level_normalize.items():
                level = re.sub(pattern, repl, level, flags=re.IGNORECASE)
            if re.match(r'^[1-5]00$', level):
                dm.level = level

        # SEMESTER: accept "1" or "2" strictly
        if semester.strip() in ['1', '2']:
            dm.semester = semester.strip()

        # course code/number primarily from course_folder
        code, num = self._course_from_text(course_folder) or (None, None)
        if not code and not num:
            code, num = self._course_from_text(stem) or (None, None)
        if code:
            dm.course_code = code.upper()
            dm.department = dm.department or code.upper()
        if num:
            dm.course_number = num
            if not dm.level and len(num) >= 3:
                derived = num[0] + '00'
                if re.match(r'^[1-5]00$', derived):
                    dm.level = derived

        # category tag
        if course_folder.upper() == 'GENERAL':
            dm.tags['category'] = 'GENERAL'
        if course_folder.upper() in {'PQ','PQS','PASTQUESTIONS','PAST QUESTIONS'} or 'PQ' in filename.upper():
            dm.tags['category'] = 'PQ'

        dm.tags['filename'] = filename
        dm.tags['stem'] = stem
        if course_folder:
            dm.tags['course_folder'] = course_folder

    def _course_from_text(self, text: str) -> Optional[Tuple[str, str]]:
        if not text:
            return None
        m = self.course_pattern.search(text)
        if not m:
            return None
        return m.group(1).replace("'S", "").replace("'s", ""), m.group(2)

    def _extract_content_metadata(self, content: str, dm: DocumentMetadata):
        dm.total_characters = len(content)
        dm.word_count = len(content.split())
        cl = content.lower()
        for pattern, d in self.document_type_patterns.items():
            if re.search(pattern, cl, re.IGNORECASE):
                dm.document_type = d
                break
        # coarse language detection
        common = [' the ', ' and ', ' or ', ' but ', ' of ', ' with ', ' for ', ' to ']
        dm.language = 'english' if sum(w in cl for w in common) >= 3 else 'unknown'
        # simple readability proxy
        import re as _re
        sentences = [s for s in _re.split(r'[.!?]+', content) if s.strip()]
        words = content.split()
        if sentences and words:
            asl = len(words) / max(1, len(sentences))
            awl = sum(len(w) for w in words) / max(1, len(words))
            score = (asl * 0.5) + (awl * 2.0)
            dm.readability_score = max(0.0, min(100.0, 100.0 - score))

    def _integrate_pdf_analysis(self, analysis: Dict, dm: DocumentMetadata):
        if 'pages' in analysis:
            dm.page_count = analysis['pages']
        if 'avg_image_area_ratio' in analysis:
            dm.image_ratio = analysis['avg_image_area_ratio']
        if 'total_chars_text_layer' in analysis:
            dm.total_characters = analysis['total_chars_text_layer']
            if dm.page_count:
                dm.text_density = dm.total_characters / dm.page_count
        dm.processing_method = 'ocr' if dm.image_ratio > 0.75 else ('direct_text' if dm.image_ratio < 0.25 else 'hybrid')

    def _group_key(self, dm: DocumentMetadata) -> str:
        if dm.tags.get('category') == 'GENERAL':
            return dm.department or 'UNKNOWN'
        if dm.department and dm.course_code and dm.course_number:
            explicit = dm.tags.get('explicit_department', False)
            if dm.department == dm.course_code and not explicit:
                return f"{dm.department}-{dm.course_number}"
            return f"{dm.department}-{dm.course_code}-{dm.course_number}"
        if dm.course_code and dm.course_number:
            return f"{dm.course_code}-{dm.course_number}"
        if dm.department and dm.course_number:
            return f"{dm.department}-{dm.course_number}"
        if dm.department and dm.course_code:
            return f"{dm.department}-{dm.course_code}"
        return dm.department or dm.course_code or 'UNKNOWN'

    def _calculate_quality(self, dm: DocumentMetadata):
        score = 0.0; maxs = 0.0
        if dm.file_hash != 'unknown': score += 10
        maxs += 10
        # PDF bits
        if dm.pdf_title: score += 10
        if dm.pdf_creation_date: score += 10
        if dm.pdf_modification_date: score += 10
        maxs += 30
        # academic
        for f in [dm.department, dm.course_code, dm.level, dm.semester]:
            if f: score += 10
            maxs += 10
        # content presence
        if dm.total_characters > 0: score += 10
        if dm.document_type: score += 10
        maxs += 20
        dm.completeness_score = (score / maxs * 100) if maxs > 0 else 0.0

    def export_metadata(self, dm: DocumentMetadata) -> Dict[str, Any]:
        data = asdict(dm)
        # ensure semester is present; author/subject/creator/producer are absent by design
        return data