# CourseGen Cloud Functions

Automatic question statistics tracking for Firestore.

## Functions

### `updateQuestionStats`
Triggered when a question is added to `Questions/{courseCode}/questions/{questionId}`

**Updates:**
- `total_questions` - Total count
- `theory_questions` - Theory question count
- `calculation_questions` - Calculation question count
- `difficulty_breakdown` - Count by Easy/Medium/Hard
- `type_difficulty_breakdown` - Count by type × difficulty

### `decrementQuestionStats`
Triggered when a question is deleted. Decrements all relevant counters.

## Quick Start

```bash
# Install Firebase CLI
npm install -g firebase-tools

# Login
firebase login

# Install dependencies
cd firebase_functions
npm install

# Deploy
./deploy.sh

# Or manually
firebase deploy --only functions
```

## Testing Locally

```bash
# Start emulator
firebase emulators:start --only functions,firestore

# In another terminal, generate a test question
python -m services.QuestionRag.pipelines.question_generator \
  --course-code "TEST 101" \
  --theory-per-request 1
```

## Monitoring

```bash
# View logs
firebase functions:log

# Filter by function
firebase functions:log --only updateQuestionStats

# Real-time logs
firebase functions:log --follow
```

## Structure

```
firebase_functions/
├── index.js           # Cloud Functions code
├── package.json       # Dependencies
├── deploy.sh          # Deployment script
└── README.md          # This file
```

## Statistics Schema

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
    theory: {
      Easy: 35,
      Medium: 45,
      Hard: 20
    },
    calculation: {
      Easy: 15,
      Medium: 25,
      Hard: 10
    }
  },
  last_updated: Timestamp,
  created_at: Timestamp
}
```

## Troubleshooting

**Functions not triggering:**
- Check deployment: `firebase functions:list`
- Verify Firestore path matches: `Questions/{courseCode}/questions/{questionId}`
- Check logs: `firebase functions:log`

**Stats incorrect:**
- Verify transaction completed: Check logs for errors
- Manually verify: Count questions in subcollection vs stats document
- Reset if needed: Delete stats document and regenerate questions

## Security

Functions use Firebase Admin SDK with elevated privileges. Ensure:
- Service account has Firestore read/write access
- Firestore rules prevent direct writes to stats documents
- Only authenticated services can write questions

## Cost

Cloud Functions pricing:
- First 2M invocations/month: Free
- Additional: $0.40 per million invocations
- Typical usage: ~2 invocations per question (create + stats update)

For 10,000 questions/month: Well within free tier.
