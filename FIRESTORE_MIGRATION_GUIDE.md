# Firestore Migration Guide

## Overview

This guide covers the migration from flat question storage to a hierarchical structure with automatic statistics tracking.

## Changes Made

### 1. **New Firestore Structure**

**Old Structure:**
```
Questions/
  {auto_id}/
    - course_code: "EEE 315"
    - question: "..."
    - ...
```

**New Structure:**
```
Questions/
  {course_code}/                    # Document (e.g., "EEE 315")
    - course_code: "EEE 315"
    - course_name: "Circuit Analysis"
    - total_questions: 150
    - theory_questions: 100
    - calculation_questions: 50
    - difficulty_breakdown:
        Easy: 50
        Medium: 70
        Hard: 30
    - type_difficulty_breakdown:
        theory:
          Easy: 35
          Medium: 45
          Hard: 20
        calculation:
          Easy: 15
          Medium: 25
          Hard: 10
    - last_updated: timestamp
    - created_at: timestamp
    
    questions/                      # Subcollection
      {auto_id}/
        - course_code: "EEE 315"
        - question: "..."
        - question_type: "theory"
        - difficulty: "Medium"
        - ...
```

### 2. **Updated Firebase Service**

**File:** `services/Firestore/firebase_service.py`

- `set_question()` now stores questions in: `Questions/{course_code}/questions/{auto_id}`
- Validates course_code before storage
- Maintains backward compatibility with Question model

### 3. **Cloud Functions**

**Location:** `firebase_functions/`

Two Cloud Functions automatically track statistics:

#### `updateQuestionStats`
- Triggered when a question is **added**
- Increments counters atomically using transactions
- Updates:
  - `total_questions`
  - `theory_questions` / `calculation_questions`
  - `difficulty_breakdown` (Easy/Medium/Hard)
  - `type_difficulty_breakdown` (theory/calculation × difficulty)

#### `decrementQuestionStats`
- Triggered when a question is **deleted**
- Decrements counters (never below 0)
- Maintains consistency

---

## Deployment Steps

### Step 1: Install Firebase CLI

```bash
npm install -g firebase-tools
firebase login
```

### Step 2: Initialize Firebase Project

```bash
cd /path/to/COURSEGEN
firebase init functions

# Select:
# - Use an existing project: [your-project-id]
# - Language: JavaScript
# - ESLint: No (optional)
# - Install dependencies: Yes
```

### Step 3: Copy Cloud Functions

```bash
# The functions are already in firebase_functions/
cd firebase_functions
npm install
```

### Step 4: Deploy Cloud Functions

```bash
# Deploy all functions
firebase deploy --only functions

# Or deploy specific function
firebase deploy --only functions:updateQuestionStats
firebase deploy --only functions:decrementQuestionStats
```

### Step 5: Verify Deployment

```bash
# Check function logs
firebase functions:log

# Test by generating a question
python -m services.QuestionRag.pipelines.question_generator \
  --course-code "EEE 315" \
  --theory-per-request 1 \
  --calc-per-request 1
```

---

## Migration Strategy

### Option 1: Fresh Start (Recommended)

1. Deploy Cloud Functions
2. Start generating questions with new structure
3. Old questions remain in flat structure (read-only)
4. Gradually migrate old questions if needed

### Option 2: Migrate Existing Questions

Create a migration script:

```python
from services.Firestore.firebase_service import FireStore
from google.cloud import firestore

def migrate_questions():
    store = FireStore()
    db = store.db
    
    # Get all old questions
    old_questions = db.collection("Questions").stream()
    
    batch = db.batch()
    count = 0
    
    for doc in old_questions:
        data = doc.to_dict()
        course_code = data.get("course_code")
        
        if not course_code:
            continue
        
        # Write to new structure
        new_ref = db.collection("Questions").document(course_code).collection("questions").document()
        batch.set(new_ref, data)
        
        count += 1
        
        # Commit in batches of 500
        if count % 500 == 0:
            batch.commit()
            batch = db.batch()
            print(f"Migrated {count} questions...")
    
    # Commit remaining
    if count % 500 != 0:
        batch.commit()
    
    print(f"Migration complete: {count} questions")

if __name__ == "__main__":
    migrate_questions()
```

**⚠️ CAUTION:** Test migration on a small dataset first!

---

## Querying Questions

### Get All Questions for a Course

```python
from services.Firestore.firebase_service import FireStore

store = FireStore()
course_code = "EEE 315"

# Get questions
questions_ref = store.db.collection("Questions").document(course_code).collection("questions")
questions = questions_ref.stream()

for q in questions:
    print(q.to_dict())
```

### Get Course Statistics

```python
from services.Firestore.firebase_service import FireStore

store = FireStore()
course_code = "EEE 315"

# Get stats
stats_ref = store.db.collection("Questions").document(course_code)
stats = stats_ref.get()

if stats.exists:
    data = stats.to_dict()
    print(f"Total Questions: {data['total_questions']}")
    print(f"Theory: {data['theory_questions']}")
    print(f"Calculation: {data['calculation_questions']}")
    print(f"Difficulty Breakdown: {data['difficulty_breakdown']}")
```

### Query by Difficulty

```python
# Get all Medium difficulty questions for a course
questions_ref = store.db.collection("Questions").document("EEE 315").collection("questions")
medium_questions = questions_ref.where("difficulty", "==", "Medium").stream()
```

### Query by Type

```python
# Get all calculation questions
calc_questions = questions_ref.where("question_type", "==", "calculation").stream()
```

---

## Monitoring

### Check Cloud Function Logs

```bash
# Real-time logs
firebase functions:log --only updateQuestionStats

# Filter by severity
firebase functions:log --only updateQuestionStats --severity ERROR
```

### Verify Statistics

```python
from services.Firestore.firebase_service import FireStore

def verify_stats(course_code):
    store = FireStore()
    
    # Get stats document
    stats_ref = store.db.collection("Questions").document(course_code)
    stats = stats_ref.get().to_dict()
    
    # Count actual questions
    questions_ref = stats_ref.collection("questions")
    actual_count = len(list(questions_ref.stream()))
    
    print(f"Stats say: {stats['total_questions']}")
    print(f"Actual count: {actual_count}")
    print(f"Match: {stats['total_questions'] == actual_count}")

verify_stats("EEE 315")
```

---

## Rollback Plan

If issues occur:

1. **Disable Cloud Functions:**
   ```bash
   firebase functions:delete updateQuestionStats
   firebase functions:delete decrementQuestionStats
   ```

2. **Revert Firebase Service:**
   ```bash
   git checkout HEAD~1 services/Firestore/firebase_service.py
   ```

3. **Redeploy:**
   ```bash
   docker-compose build
   docker-compose up
   ```

---

## Security Rules

Update Firestore security rules:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Questions collection
    match /Questions/{courseCode} {
      // Allow read access to stats
      allow read: if true;
      
      // Only allow writes from Cloud Functions
      allow write: if false;
      
      // Questions subcollection
      match /questions/{questionId} {
        allow read: if true;
        // Only allow writes from authenticated services
        allow write: if request.auth != null;
      }
    }
  }
}
```

---

## Testing

### Local Testing with Emulator

```bash
cd firebase_functions
npm install
firebase emulators:start --only functions,firestore

# In another terminal, run your question generation
python -m services.QuestionRag.pipelines.question_generator \
  --course-code "TEST 101" \
  --theory-per-request 2
```

### Unit Test Cloud Function

```javascript
// firebase_functions/test/index.test.js
const test = require('firebase-functions-test')();
const myFunctions = require('../index');

describe('updateQuestionStats', () => {
  it('should increment counters', async () => {
    const snap = test.firestore.makeDocumentSnapshot(
      {
        course_code: 'EEE 315',
        question_type: 'theory',
        difficulty: 'Medium'
      },
      'Questions/EEE 315/questions/test123'
    );
    
    const wrapped = test.wrap(myFunctions.updateQuestionStats);
    await wrapped(snap, { params: { courseCode: 'EEE 315' } });
    
    // Verify stats were updated
  });
});
```

---

## Troubleshooting

### Issue: Stats not updating

**Check:**
1. Cloud Functions deployed: `firebase functions:list`
2. Function logs: `firebase functions:log`
3. Firestore permissions
4. Service account has Firestore access

### Issue: Duplicate counts

**Solution:**
- Cloud Functions are idempotent
- If duplicate triggers occur, manually reset stats:

```python
store.db.collection("Questions").document("EEE 315").set({
    "total_questions": 0,
    "theory_questions": 0,
    "calculation_questions": 0,
    # ... reset all fields
})
```

Then regenerate questions.

---

## Benefits

✅ **Organized Structure:** Questions grouped by course code  
✅ **Automatic Statistics:** Real-time tracking without manual updates  
✅ **Scalable:** Subcollections handle unlimited questions per course  
✅ **Atomic Updates:** Transactions prevent race conditions  
✅ **Query Efficiency:** Filter by course, type, difficulty  
✅ **Backward Compatible:** Old questions remain accessible  

---

## Support

For issues or questions:
1. Check Cloud Function logs: `firebase functions:log`
2. Verify Firestore structure in Firebase Console
3. Test with a single question first
4. Review transaction logs for conflicts
