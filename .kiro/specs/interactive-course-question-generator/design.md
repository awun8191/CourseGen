# Design Document

## Overview

The Interactive Course Question Generator is an enhancement to the existing RAG pipeline that provides a terminal-based interface for course-specific question generation. The system allows educators to input course codes, automatically retrieve course metadata, generate course topics with descriptions, and create targeted questions for engineering students. The design emphasizes the critical workflow of generating topics first before any question generation occurs.

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Interactive    │    │   Course Topic   │    │   Question      │
│  Terminal       │───▶│   Generator      │───▶│   Generator     │
│  Interface      │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  courses.json   │    │   Gemini 2.5     │    │   ChromaDB      │
│  Lookup         │    │   Flash Lite     │    │   RAG Context   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Component Architecture

The system integrates with existing infrastructure while adding new interactive capabilities:

1. **Interactive Terminal Module**: New command-line interface
2. **Course Lookup Service**: Enhanced courses.json integration
3. **Topic Generation Service**: New Gemini-powered topic creation
4. **Enhanced Question Generator**: Extended existing question generation
5. **Context Selection Engine**: New RAG vs Gemini-only routing

## Components and Interfaces

### 1. Interactive Terminal Interface

**Location**: `services/InteractiveCourse/terminal_interface.py`

**Responsibilities**:
- Display welcome message and course code prompts
- Handle user input validation and error messages
- Present context selection options (RAG vs Gemini-only)
- Display generated topics for selection
- Manage session flow and exit conditions

**Key Methods**:
```python
class InteractiveTerminal:
    def start_session(self) -> None
    def prompt_course_code(self) -> str
    def display_course_info(self, course_data: CourseData) -> None
    def select_context_option(self) -> ContextOption
    def display_topics(self, topics: List[Topic]) -> None
    def select_topics(self, topics: List[Topic]) -> List[Topic]
    def handle_exit(self) -> None
```

### 2. Course Lookup Service

**Location**: `services/InteractiveCourse/course_lookup.py`

**Responsibilities**:
- Search courses.json for course code matches
- Retrieve course metadata (title, departments, level, semester)
- Handle course code validation and error cases
- Provide course data in structured format

**Key Methods**:
```python
class CourseLookupService:
    def __init__(self, courses_file_path: str)
    def find_course(self, course_code: str) -> Optional[CourseData]
    def validate_course_code(self, course_code: str) -> bool
    def get_department_courses(self, department: str, semester: str) -> List[CourseData]
```

### 3. Topic Generation Service

**Location**: `services/InteractiveCourse/topic_generator.py`

**Responsibilities**:
- Check for existing course topics/outlines in courses.json before generation
- Generate course topics using Gemini 2.5 Flash Lite only if not cached
- Limit topics to maximum of 10 per course
- Create detailed descriptions for each topic
- Store topics and descriptions in courses.json to prevent regeneration
- Handle topic generation failures and retries
- Use existing ApiKeyManager for intelligent key rotation

**Key Methods**:
```python
class TopicGeneratorService:
    def __init__(self, gemini_service: GeminiService)
    def generate_topics(self, course_data: CourseData) -> List[Topic]
    def check_existing_topics(self, course_code: str) -> Optional[List[Topic]]
    def store_topics_in_courses_json(self, course_code: str, topics: List[Topic]) -> bool
    def validate_topic_count(self, topics: List[Topic]) -> bool
    def create_topic_descriptions(self, course_data: CourseData, topic_names: List[str]) -> List[Topic]
```

### 4. Context Selection Engine

**Location**: `services/InteractiveCourse/context_selector.py`

**Responsibilities**:
- Route between RAG-enhanced and Gemini-only question generation
- Query ChromaDB for department/semester-specific content
- Prepare context data for question generation
- Handle context retrieval failures

**Key Methods**:
```python
class ContextSelector:
    def __init__(self, chroma_service: ChromaDBService)
    def select_rag_context(self, course_data: CourseData) -> RAGContext
    def select_gemini_only_context(self, course_data: CourseData) -> GeminiContext
    def query_department_semester_data(self, department: str, semester: str) -> List[Document]
```

### 5. Enhanced Question Generator

**Location**: `services/InteractiveCourse/question_generator.py`

**Responsibilities**:
- Generate questions for selected topics using chosen context approach
- Integrate with existing Gemini question generation parameters
- Support batch processing for multiple topics
- Include comprehensive metadata in generated questions

**Key Methods**:
```python
class EnhancedQuestionGenerator:
    def __init__(self, gemini_service: GeminiService, context_selector: ContextSelector)
    def generate_questions_with_rag(self, topics: List[Topic], rag_context: RAGContext) -> List[Question]
    def generate_questions_gemini_only(self, topics: List[Topic], gemini_context: GeminiContext) -> List[Question]
    def batch_generate_questions(self, topics: List[Topic], context_option: ContextOption) -> List[Question]
    def _initialize_gemini_with_load_balancer(self) -> GeminiService
```

## Data Models

### Course Data Model

**Location**: `data_models/interactive_course.py`

```python
@dataclass
class CourseData:
    code: str
    title: str
    offered_by_programs: List[str]
    type: str
    units: int
    levels: List[str]
    semesters: List[str]
    is_elective: bool
    topics: Optional[List[Topic]] = None

@dataclass
class Topic:
    name: str
    description: str
    course_code: str
    generated_timestamp: datetime

@dataclass
class Question:
    content: str
    topic: str
    course_code: str
    department: str
    semester: str
    level: str
    generation_method: str  # "rag" or "gemini_only"
    metadata: Dict[str, Any]
    created_at: datetime
```

### Context Models

```python
@dataclass
class RAGContext:
    course_data: CourseData
    related_documents: List[Document]
    department_content: List[str]
    semester_content: List[str]

@dataclass
class GeminiContext:
    course_data: CourseData
    course_outline: str
    department_info: str

class ContextOption(Enum):
    RAG_ENHANCED = "rag"
    GEMINI_ONLY = "gemini_only"
```

## Integration with Existing Infrastructure

### Gemini Service Integration

The system leverages the existing Gemini service infrastructure with intelligent load balancing:

- **API Key Management**: Uses existing `services/Gemini/api_key_manager.py` for intelligent load balancing
- **Service Client**: Extends `services/Gemini/gemini_service.py` with existing ApiKeyManager
- **Configuration**: Utilizes existing `data_models/gemini_config.py`
- **Usage Tracking**: Stores API key usage in `services/data/gemini_cache/api_key_cache.json`
- **Smart Rotation**: API keys rotate only when reaching individual limits (not round-robin)
- **Caching**: Integrates with existing Gemini response caching and course outline caching
- **Service Path**: Uses services from `/home/raregazetto/Documents/Recursive-PDF-EXTRACTION-AND-RAG/COURSEGEN/services/Gemini/`

### ChromaDB Integration

RAG context retrieval uses existing ChromaDB infrastructure:

- **Query Interface**: Uses `services/QuestionRag/chromadb_query.py`
- **BGE-M3 Embeddings**: Leverages existing Cloudflare embeddings via `services/Cloudflare/`
- **Metadata Filtering**: Extends existing metadata queries for department/semester filtering

### Configuration Integration

The system uses existing configuration patterns:

- **Environment Variables**: Follows existing patterns for API keys and settings
- **Config Files**: Integrates with existing `config.py` structure
- **Logging**: Uses existing `utils/logging_utils.py` for colored output

## Workflow Implementation

### Primary Workflow

1. **Session Initialization**
   - Display welcome message
   - Initialize services (Gemini, ChromaDB, Course Lookup)
   - Load courses.json data

2. **Course Code Input**
   - Prompt for course code
   - Validate format and existence
   - Display course information
   - Handle invalid codes with retry

3. **Context Selection**
   - Present two options: RAG-enhanced or Gemini-only
   - Explain each option's benefits
   - Capture user selection

4. **Topic Generation** (Critical First Step)
   - Check if course outline/topics already exist in courses.json
   - If not found, use Gemini 2.5 Flash Lite with course metadata
   - Generate maximum 10 topics with descriptions
   - Store topics in courses.json with metadata to prevent regeneration
   - Display topics to user for selection

5. **Topic Selection**
   - Present generated topics in numbered list
   - Allow multiple topic selection
   - Support "all topics" option

6. **Question Generation**
   - Route to appropriate generation method based on context selection
   - Generate questions for selected topics
   - Include comprehensive metadata
   - Display generation progress

7. **Output and Storage**
   - Save questions in specified formats (JSON, CSV, JSONL)
   - Display summary statistics
   - Offer to continue with another course

### Error Handling Strategy

- **Course Not Found**: Clear error message with suggestion to check course code
- **Topic Generation Failure**: Retry mechanism with fallback to manual topic input
- **API Failures**: Graceful degradation with informative error messages
- **Storage Failures**: Retry with alternative storage locations
- **Network Issues**: Timeout handling with user notification

## Testing Strategy

### Unit Testing

- **Course Lookup Service**: Test course code validation and retrieval
- **Topic Generator**: Test topic generation limits and description quality
- **Context Selector**: Test RAG vs Gemini-only routing logic
- **Question Generator**: Test question generation with both context types

### Integration Testing

- **End-to-End Workflow**: Test complete user journey from course input to question output
- **Gemini Integration**: Test API calls and response handling
- **ChromaDB Integration**: Test RAG context retrieval and filtering
- **File Operations**: Test courses.json reading and writing

### User Acceptance Testing

- **Terminal Interface**: Test user experience and input validation
- **Topic Quality**: Validate generated topics are relevant and comprehensive
- **Question Quality**: Ensure questions are appropriate for course level and topics
- **Performance**: Test response times for topic and question generation

## Performance Considerations

### Caching Strategy

- **Course Data**: Cache loaded courses.json in memory
- **Generated Topics**: Cache topics to avoid regeneration
- **Gemini Responses**: Leverage existing Gemini caching
- **ChromaDB Queries**: Cache department/semester query results

### Optimization Targets

- **Topic Generation**: < 30 seconds for 10 topics
- **Question Generation**: < 2 minutes for 5 questions per topic
- **Course Lookup**: < 1 second for course code search
- **Context Retrieval**: < 10 seconds for RAG context assembly

### Scalability Considerations

- **Concurrent Sessions**: Support multiple terminal sessions
- **Batch Processing**: Efficient handling of multiple topic selections
- **Memory Management**: Proper cleanup of large context data
- **API Rate Limiting**: Respect Gemini API limits with queuing

## Security and Data Privacy

### Data Handling

- **Course Data**: Read-only access to courses.json with backup creation
- **Generated Content**: Secure storage with appropriate file permissions
- **API Keys**: Use existing secure key management patterns
- **User Input**: Sanitize and validate all terminal inputs

### Privacy Considerations

- **No Personal Data**: System processes only academic course information
- **Audit Logging**: Log user actions for debugging without sensitive data
- **Data Retention**: Clear guidelines for generated question storage
- **Access Control**: Terminal access follows system user permissions