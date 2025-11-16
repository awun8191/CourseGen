# Firestore Quick Reference

## 🚀 Quick Deploy

```bash
# 1. Deploy Cloud Functions
cd firebase_functions
npm install
./deploy.sh

# 2. Verify deployment
firebase functions:list

# 3. Test with sample course
docker-compose run --rm coursegen \
  --course-code "TEST 101" \
  --theory-per-request 2 \
  --calc-per-request 2

# 4. Verify structure
python verify_firestore_structure.py
```

---

## 📊 New Structure

```
Questions/
  EEE 315/                    # Course document with stats
    ├── total_questions: 150
    ├── theory_questions: 100
    ├── calculation_questions: 50
    └── questions/            # Subcollection
          ├── abc123/         # Individual questions
          ├── def456/
          └── ...
```

---

## 🔍 Common Queries

### Get Course Stats
```python
from services.Firestore.firebase_service import FireStore

store = FireStore()
stats = store.db.collection("Questions").document("EEE 315").get()
print(stats.to_dict())
```

### Get All Questions for Course
```python
questions = store.db.collection("Questions") \
    .document("EEE 315") \
    .collection("questions") \
    .stream()
```

### Filter by Type
```python
theory_questions = store.db.collection("Questions") \
    .document("EEE 315") \
    .collection("questions") \
    .where("question_type", "==", "theory") \
    .stream()
```

### Filter by Difficulty
```python
hard_questions = store.db.collection("Questions") \
    .document("EEE 315") \
    .collection("questions") \
    .where("difficulty", "==", "Hard") \
    .stream()
```

---

## 🔧 Troubleshooting

### Stats not updating?
```bash
# Check functions deployed
firebase functions:list

# Check logs
firebase functions:log --only updateQuestionStats

# Verify path in code
# Should be: Questions/{courseCode}/questions/{questionId}
```

### Stats mismatch?
```bash
# Run verification
python verify_firestore_structure.py

# Check for errors
firebase functions:log --severity ERROR
```

### Need to reset stats?
```python
# Delete stats document (questions remain)
store.db.collection("Questions").document("EEE 315").delete()

# Regenerate questions to rebuild stats
```

---

## 📝 Monitoring

```bash
# Real-time logs
firebase functions:log --follow

# Specific function
firebase functions:log --only updateQuestionStats

# Errors only
firebase functions:log --severity ERROR
```

---

## 🎯 Statistics Schema

```javascript
{
  course_code: "EEE 315",
  course_name: "Circuit Analysis",
  total_questions: 150,
  theory_questions: 100,
  calculation_questions: 50,
  difficulty_breakdown: {
    Easy: 50,
    Medium: 70,
    Hard: 30
  },
  type_difficulty_breakdown: {
    theory: { Easy: 35, Medium: 45, Hard: 20 },
    calculation: { Easy: 15, Medium: 25, Hard: 10 }
  },
  last_updated: Timestamp,
  created_at: Timestamp
}
```

---

## ⚡ Key Commands

| Action | Command |
|--------|---------|
| Deploy functions | `cd firebase_functions && ./deploy.sh` |
| List functions | `firebase functions:list` |
| View logs | `firebase functions:log` |
| Verify structure | `python verify_firestore_structure.py` |
| Generate questions | `docker-compose run --rm coursegen --course-code "EEE 315"` |
| Delete function | `firebase functions:delete updateQuestionStats` |

---

## 🔐 Security Rules

```javascript
match /Questions/{courseCode} {
  allow read: if true;
  allow write: if false;  // Only Cloud Functions
  
  match /questions/{questionId} {
    allow read: if true;
    allow write: if request.auth != null;
  }
}
```

---

## 💰 Cost Estimate

- **Cloud Functions:** Free tier (2M invocations/month)
- **Firestore:** ~$0.036 per 10K questions
- **Total:** Essentially free for typical usage ✅

---

## 📚 Documentation

- Full guide: `FIRESTORE_MIGRATION_GUIDE.md`
- Summary: `FIRESTORE_UPDATE_SUMMARY.md`
- Functions: `firebase_functions/README.md`
