# Firestore Update Summary

## ✅ Implementation Complete

The Firestore storage has been restructured with automatic statistics tracking via Cloud Functions.

---

## Changes Made

### 1. **Firebase Service Update** ✅

**File:** `services/Firestore/firebase_service.py`

**Change:** `set_question()` method now stores questions hierarchically:

```python
# OLD: Questions/{auto_id}
# NEW: Questions/{course_code}/questions/{auto_id}
```

**Benefits:**
- Questions organized by course code
- Easy to query all questions for a specific course
- Scalable subcollection structure
- Maintains backward compatibility with Question model

### 2. **Cloud Functions Created** ✅

**Location:** `firebase_functions/`

**Files:**
- `index.js` - Cloud Functions implementation
- `package.json` - Dependencies
- `deploy.sh` - Deployment script
- `README.md` - Documentation
- `.gitignore` - Ignore node_modules

**Functions:**

#### `updateQuestionStats`
- **Trigger:** Question added to `Questions/{courseCode}/questions/{questionId}`
- **Action:** Atomically increments statistics using Firestore transactions
- **Updates:**
  - `total_questions`
  - `theory_questions` / `calculation_questions`
  - `difficulty_breakdown` (Easy/Medium/Hard counts)
  - `type_difficulty_breakdown` (theory/calculation × difficulty matrix)
  - `last_updated` timestamp

#### `decrementQuestionStats`
- **Trigger:** Question deleted from subcollection
- **Action:** Atomically decrements statistics (never below 0)
- **Ensures:** Consistency when questions are removed

### 3. **Documentation Created** ✅

**Files:**
- `FIRESTORE_MIGRATION_GUIDE.md` - Complete migration guide
- `firebase_functions/README.md` - Cloud Functions documentation
- `FIRESTORE_UPDATE_SUMMARY.md` - This file

### 4. **Verification Script** ✅

**File:** `verify_firestore_structure.py`

**Features:**
- Lists all courses with questions
- Verifies statistics accuracy
- Compares stats document vs actual question counts
- Identifies mismatches

---

## New Firestore Structure

```
Questions/
  {course_code}/                          # Document (e.g., "EEE 315")
    ├── course_code: "EEE 315"
    ├── course_name: "Circuit Analysis"
    ├── total_questions: 150
    ├── theory_questions: 100
    ├── calculation_questions: 50
    ├── difficulty_breakdown:
    │     ├── Easy: 50
    │     ├── Medium: 70
    │     └── Hard: 30
    ├── type_difficulty_breakdown:
    │     ├── theory:
    │     │     ├── Easy: 35
    │     │     ├── Medium: 45
    │     │     └── Hard: 20
    │     └── calculation:
    │           ├── Easy: 15
    │           ├── Medium: 25
    │           └── Hard: 10
    ├── last_updated: Timestamp
    └── created_at: Timestamp
    
    questions/                            # Subcollection
      ├── {auto_id_1}/
      │     ├── course_code: "EEE 315"
      │     ├── question: "..."
      │     ├── question_type: "theory"
      │     ├── difficulty: "Medium"
      │     └── ... (all Question model fields)
      ├── {auto_id_2}/
      └── ...
```

---

## Deployment Steps

### Prerequisites

```bash
# Install Firebase CLI
npm install -g firebase-tools

# Login to Firebase
firebase login
```

### Deploy Cloud Functions

```bash
cd firebase_functions

# Install dependencies
npm install

# Deploy
./deploy.sh

# Or manually
firebase deploy --only functions
```

### Verify Deployment

```bash
# List deployed functions
firebase functions:list

# Expected output:
# ✔ updateQuestionStats
# ✔ decrementQuestionStats

# Check logs
firebase functions:log
```

---

## Testing

### 1. Generate Test Questions

```bash
# Generate a few questions for testing
docker-compose run --rm coursegen \
  --course-code "TEST 101" \
  --theory-per-request 2 \
  --calc-per-request 2
```

### 2. Verify Structure

```bash
# Run verification script
python verify_firestore_structure.py
```

**Expected Output:**
```
📚 Courses with Questions:
  TEST 101        | Total:    4 | Theory:    2 | Calc:    2

📊 Verifying statistics for: TEST 101
  Total Questions:
    Stats:  4
    Actual: 4
    ✅ Match
  
  Theory Questions:
    Stats:  2
    Actual: 2
    ✅ Match
  
  Calculation Questions:
    Stats:  2
    Actual: 2
    ✅ Match
```

### 3. Check Firebase Console

1. Go to Firebase Console → Firestore Database
2. Navigate to `Questions` collection
3. You should see documents named by course code (e.g., "TEST 101")
4. Click on a course document to see statistics
5. Click on `questions` subcollection to see individual questions

---

## Usage Examples

### Query Questions for a Course

```python
from services.Firestore.firebase_service import FireStore

store = FireStore()
course_code = "EEE 315"

# Get all questions
questions_ref = store.db.collection("Questions").document(course_code).collection("questions")
questions = questions_ref.stream()

for q in questions:
    data = q.to_dict()
    print(f"{data['question_type']}: {data['question'][:50]}...")
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
    print(f"Course: {data['course_name']}")
    print(f"Total: {data['total_questions']}")
    print(f"Theory: {data['theory_questions']}")
    print(f"Calculation: {data['calculation_questions']}")
    print(f"Difficulty: {data['difficulty_breakdown']}")
```

### Query by Type and Difficulty

```python
# Get all Medium difficulty theory questions
questions_ref = store.db.collection("Questions").document("EEE 315").collection("questions")

medium_theory = questions_ref \
    .where("question_type", "==", "theory") \
    .where("difficulty", "==", "Medium") \
    .stream()

for q in medium_theory:
    print(q.to_dict()['question'])
```

---

## Monitoring

### Cloud Function Logs

```bash
# Real-time logs
firebase functions:log --follow

# Filter by function
firebase functions:log --only updateQuestionStats

# Filter by severity
firebase functions:log --severity ERROR
```

### Check Statistics Accuracy

```bash
# Run verification script
python verify_firestore_structure.py

# Or verify specific course
python -c "
from verify_firestore_structure import verify_course_stats
verify_course_stats('EEE 315')
"
```

---

## Migration from Old Structure

If you have existing questions in the flat structure:

### Option 1: Fresh Start (Recommended)
- Deploy Cloud Functions
- Start generating new questions
- Old questions remain accessible but separate

### Option 2: Migrate Existing Questions

**⚠️ CAUTION:** Test on a small dataset first!

```python
from services.Firestore.firebase_service import FireStore

def migrate_old_questions():
    store = FireStore()
    db = store.db
    
    # Get old questions (flat structure)
    old_questions = db.collection("Questions").stream()
    
    migrated = 0
    skipped = 0
    
    for doc in old_questions:
        data = doc.to_dict()
        
        # Skip if it's already a stats document (has total_questions field)
        if 'total_questions' in data:
            skipped += 1
            continue
        
        course_code = data.get("course_code")
        if not course_code:
            skipped += 1
            continue
        
        # Write to new structure
        new_ref = db.collection("Questions") \
                    .document(course_code) \
                    .collection("questions") \
                    .document()
        new_ref.set(data)
        
        migrated += 1
        
        if migrated % 100 == 0:
            print(f"Migrated {migrated} questions...")
    
    print(f"✅ Migration complete: {migrated} migrated, {skipped} skipped")

# Run migration
migrate_old_questions()
```

---

## Troubleshooting

### Issue: Statistics not updating

**Symptoms:** Questions are stored but stats remain at 0

**Solutions:**
1. Check Cloud Functions are deployed:
   ```bash
   firebase functions:list
   ```

2. Check function logs for errors:
   ```bash
   firebase functions:log --only updateQuestionStats
   ```

3. Verify Firestore path matches:
   - Should be: `Questions/{courseCode}/questions/{questionId}`
   - Check your `set_question()` implementation

4. Test with Firebase emulator locally:
   ```bash
   firebase emulators:start --only functions,firestore
   ```

### Issue: Stats mismatch

**Symptoms:** Stats don't match actual question count

**Solutions:**
1. Run verification script:
   ```bash
   python verify_firestore_structure.py
   ```

2. Check for failed transactions in logs:
   ```bash
   firebase functions:log --severity ERROR
   ```

3. Manually reset stats (last resort):
   ```python
   store.db.collection("Questions").document("EEE 315").delete()
   # Then regenerate questions
   ```

### Issue: Duplicate counts

**Cause:** Function triggered multiple times

**Solution:**
- Cloud Functions are idempotent by design
- Check logs for duplicate triggers
- Verify no manual writes to stats documents

---

## Security Considerations

### Firestore Rules

Update your Firestore security rules:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /Questions/{courseCode} {
      // Stats document - read-only for clients
      allow read: if true;
      allow write: if false;  // Only Cloud Functions can write
      
      match /questions/{questionId} {
        // Questions - authenticated writes only
        allow read: if true;
        allow write: if request.auth != null;
      }
    }
  }
}
```

### Service Account Permissions

Ensure your service account has:
- `Cloud Datastore User` role (for Firestore access)
- `Cloud Functions Developer` role (for deployment)

---

## Performance & Cost

### Cloud Functions Pricing
- **Free tier:** 2M invocations/month
- **Additional:** $0.40 per million invocations
- **Typical usage:** 2 invocations per question (create + update)

**Example:**
- 10,000 questions/month = 20,000 invocations
- Well within free tier ✅

### Firestore Pricing
- **Reads:** $0.06 per 100K documents
- **Writes:** $0.18 per 100K documents
- **Storage:** $0.18 per GB/month

**Example:**
- 10,000 questions = 10,000 writes + 10,000 stats updates = 20,000 writes
- Cost: ~$0.036/month ✅

---

## Benefits

✅ **Organized:** Questions grouped by course code  
✅ **Automatic:** Real-time statistics without manual updates  
✅ **Scalable:** Subcollections handle unlimited questions  
✅ **Atomic:** Transactions prevent race conditions  
✅ **Queryable:** Filter by course, type, difficulty  
✅ **Reliable:** Cloud Functions ensure consistency  
✅ **Cost-effective:** Well within free tiers  

---

## Next Steps

1. **Deploy Cloud Functions:**
   ```bash
   cd firebase_functions && ./deploy.sh
   ```

2. **Test with sample course:**
   ```bash
   docker-compose run --rm coursegen \
     --course-code "TEST 101" \
     --theory-per-request 2 \
     --calc-per-request 2
   ```

3. **Verify structure:**
   ```bash
   python verify_firestore_structure.py
   ```

4. **Generate production questions:**
   ```bash
   docker-compose run --rm coursegen \
     --course-code "EEE 315" \
     --theory-per-request 10 \
     --calc-per-request 5
   ```

5. **Monitor logs:**
   ```bash
   firebase functions:log --follow
   ```

---

## Rollback Plan

If issues occur:

1. **Disable Cloud Functions:**
   ```bash
   firebase functions:delete updateQuestionStats
   firebase functions:delete decrementQuestionStats
   ```

2. **Revert Firebase service:**
   ```bash
   git checkout HEAD~1 services/Firestore/firebase_service.py
   ```

3. **Rebuild and redeploy:**
   ```bash
   docker-compose build
   docker-compose up
   ```

---

## Support

For issues:
1. Check Cloud Function logs: `firebase functions:log`
2. Run verification script: `python verify_firestore_structure.py`
3. Review Firebase Console for data structure
4. Check service account permissions
5. Test with Firebase emulator locally

---

**Implementation Date:** 2025-11-09  
**Status:** ✅ Ready for Deployment  
**Risk Level:** 🟢 Low (backward compatible, atomic operations, rollback available)
