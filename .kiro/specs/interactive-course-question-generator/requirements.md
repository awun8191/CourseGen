# Requirements Document

## Introduction

This feature enhances the existing question generation RAG pipeline by adding an interactive terminal interface that allows users to input course codes, retrieve course metadata from the courses.json file, generate course outlines, and create topic-specific questions for engineering students. The system will leverage the existing BGE-M3 embeddings via Cloudflare, ChromaDB storage, and Gemini 2.5 Flash Lite for question generation while maintaining critical metadata for semester, department, and course code information.

## Requirements

### Requirement 1

**User Story:** As an educator, I want to input a course code in an interactive terminal so that I can quickly access course information and generate relevant educational content with flexible context options.

#### Acceptance Criteria

1. WHEN a user launches the interactive terminal THEN the system SHALL display a welcome message and prompt for course code input
2. WHEN a user enters a course code (e.g., "AAE 101", "BME 313") THEN the system SHALL search the courses.json file for matching course data
3. WHEN a valid course code is found THEN the system SHALL display the course title, offering programs, type, units, levels, and semesters
4. WHEN course information is displayed THEN the system SHALL present two context options for question generation:
   - Option A: Use all RAG data from the same department and semester to provide rich context
   - Option B: Use only Gemini with course code, title, and offering departments for standalone generation
5. WHEN an invalid course code is entered THEN the system SHALL display an error message and prompt for re-entry
6. WHEN a user enters "exit" or "quit" THEN the system SHALL gracefully terminate the session

### Requirement 2

**User Story:** As an educator, I want the system to automatically generate course topics and descriptions first as the foundation for topic-specific question generation.

#### Acceptance Criteria

1. WHEN a context option is selected THEN the system SHALL FIRST generate course topics and descriptions before any question generation
2. WHEN generating course topics THEN the system SHALL use Gemini 2.5 Flash Lite with the course title, offering departments, and academic level to create detailed topic-based content
3. WHEN course topics are generated THEN the system SHALL structure them into logical topics with clear descriptions (no subtopics required)
4. WHEN topics are generated THEN the system SHALL limit the number of topics to a maximum of 10 topics per course
5. WHEN topics and descriptions are created THEN the system SHALL store them in the courses.json file with metadata including course code, department, semester, and level information
6. WHEN topic generation is complete THEN the system SHALL display the generated topics for user selection before proceeding to question generation
7. WHEN topic generation fails THEN the system SHALL provide fallback options or error handling

### Requirement 3

**User Story:** As an educator, I want to generate questions for individual course topics using my selected context approach so that I can create targeted assessments for specific learning objectives.

#### Acceptance Criteria

1. WHEN a course outline is generated THEN the system SHALL present topics for question generation selection
2. WHEN Option A (RAG context) is selected AND topics are chosen THEN the system SHALL query ChromaDB for all textbook content from the same department and semester using BGE-M3 embeddings
3. WHEN Option B (Gemini-only) is selected AND topics are chosen THEN the system SHALL use only the course code, title, offering departments, and generated outline to inform Gemini about question generation context
4. WHEN relevant content is retrieved (Option A) THEN the system SHALL use Gemini 2.5 Flash Lite with RAG context to generate questions specific to the selected topics
5. WHEN using Gemini-only approach (Option B) THEN the system SHALL use Gemini 2.5 Flash Lite with course metadata to generate contextually appropriate questions
6. WHEN questions are generated THEN the system SHALL include metadata for course code, department, semester, level, topic, and generation method used
7. WHEN batch processing is requested THEN the system SHALL generate questions for multiple topics efficiently using the selected context approach

### Requirement 4

**User Story:** As a system administrator, I want the question generation to integrate seamlessly with the existing RAG pipeline so that all generated content maintains consistency and quality.

#### Acceptance Criteria

1. WHEN questions are generated THEN the system SHALL use the existing BGE-M3 via Cloudflare embeddings infrastructure
2. WHEN storing questions THEN the system SHALL utilize the existing ChromaDB storage with proper metadata indexing
3. WHEN processing requests THEN the system SHALL maintain the existing caching mechanisms for OCR and API responses
4. WHEN generating content THEN the system SHALL follow the existing Gemini API key rotation and management patterns
5. WHEN errors occur THEN the system SHALL use the existing logging and error handling utilities

### Requirement 5

**User Story:** As an educator, I want to save and export generated questions so that I can use them in various educational platforms and assessments.

#### Acceptance Criteria

1. WHEN questions are generated THEN the system SHALL offer options to save questions in multiple formats (JSON, CSV, JSONL)
2. WHEN saving questions THEN the system SHALL preserve all metadata including course information, topic details, and generation timestamps
3. WHEN exporting data THEN the system SHALL organize questions by course, topic, and difficulty level
4. WHEN batch operations complete THEN the system SHALL provide summary statistics of generated content
5. WHEN storage operations fail THEN the system SHALL provide clear error messages and retry options

### Requirement 6

**User Story:** As an educator, I want to configure question generation parameters so that I can customize the output to match my specific teaching requirements.

#### Acceptance Criteria

1. WHEN using the interactive terminal THEN the system SHALL allow configuration of question types, difficulty levels, and quantity
2. WHEN generating questions THEN the system SHALL respect the existing Gemini configuration parameters in the codebase
3. WHEN processing multiple topics THEN the system SHALL allow batch configuration settings
4. WHEN parameters are invalid THEN the system SHALL provide validation feedback and default values
5. WHEN configurations are set THEN the system SHALL persist settings for the current session