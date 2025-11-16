# Firestore Deployment Checklist

## Pre-Deployment

- [ ] **Backup existing data**
  ```bash
  # Export Firestore data (optional)
  gcloud firestore export gs://your-bucket/backup-$(date +%Y%m%d)
  ```

- [ ] **Review changes**
  - [ ] Read `FIRESTORE_UPDATE_SUMMARY.md`
  - [ ] Review `firebase_service.py` changes
  - [ ] Review Cloud Functions code in `firebase_functions/index.js`

- [ ] **Test locally (optional)**
  ```bash
  cd firebase_functions
  firebase emulators:start --only functions,firestore
  ```

---

## Deployment Steps

### 1. Deploy Cloud Functions

- [ ] **Install Firebase CLI**
  ```bash
  npm install -g firebase-tools
  ```

- [ ] **Login to Firebase**
  ```bash
  firebase login
  ```

- [ ] **Deploy functions**
  ```bash
  cd firebase_functions
  npm install
  ./deploy.sh
  ```

- [ ] **Verify deployment**
  ```bash
  firebase functions:list
  # Should show: updateQuestionStats, decrementQuestionStats
  ```

### 2. Update Application Code

- [ ] **Rebuild Docker image**
  ```bash
  cd ..
  ./build.sh --cleanup
  ```

- [ ] **Deploy to ECR (if using AWS)**
  ```bash
  ./build.sh --deploy
  ```

### 3. Test Deployment

- [ ] **Generate test questions**
  ```bash
  docker-compose run --rm coursegen \
    --course-code "TEST 101" \
    --theory-per-request 2 \
    --calc-per-request 2
  ```

- [ ] **Verify structure in Firebase Console**
  - Navigate to Firestore Database
  - Check `Questions/TEST 101` document exists
  - Verify stats fields are populated
  - Check `questions` subcollection has 4 questions

- [ ] **Run verification script**
  ```bash
  python verify_firestore_structure.py
  # Should show all stats matching actual counts
  ```

- [ ] **Check Cloud Function logs**
  ```bash
  firebase functions:log --only updateQuestionStats
  # Should show successful executions
  ```

### 4. Production Deployment

- [ ] **Generate questions for real course**
  ```bash
  docker-compose run --rm coursegen \
    --course-code "EEE 315" \
    --theory-per-request 10 \
    --calc-per-request 5
  ```

- [ ] **Monitor logs during generation**
  ```bash
  # In another terminal
  firebase functions:log --follow
  ```

- [ ] **Verify statistics accuracy**
  ```bash
  python verify_firestore_structure.py
  ```

---

## Post-Deployment

### Monitoring

- [ ] **Set up log monitoring**
  ```bash
  # Check logs daily for first week
  firebase functions:log --severity ERROR
  ```

- [ ] **Verify statistics weekly**
  ```bash
  python verify_firestore_structure.py
  ```

### Security

- [ ] **Update Firestore security rules**
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

- [ ] **Verify service account permissions**
  - Cloud Datastore User
  - Cloud Functions Developer

### Documentation

- [ ] **Update team documentation**
  - Share `FIRESTORE_QUICK_REFERENCE.md`
  - Document new query patterns
  - Update API documentation if applicable

---

## Rollback Plan (If Needed)

- [ ] **Disable Cloud Functions**
  ```bash
  firebase functions:delete updateQuestionStats
  firebase functions:delete decrementQuestionStats
  ```

- [ ] **Revert code changes**
  ```bash
  git checkout HEAD~1 services/Firestore/firebase_service.py
  ```

- [ ] **Rebuild and redeploy**
  ```bash
  ./build.sh --cleanup
  docker-compose up
  ```

---

## Success Criteria

✅ **Cloud Functions deployed and listed**  
✅ **Test questions generated successfully**  
✅ **Statistics match actual question counts**  
✅ **No errors in Cloud Function logs**  
✅ **Production questions generated successfully**  
✅ **Team trained on new structure**  

---

## Troubleshooting Contacts

| Issue | Action |
|-------|--------|
| Cloud Functions not deploying | Check Firebase CLI version, re-login |
| Stats not updating | Check function logs, verify Firestore path |
| Permission errors | Verify service account roles |
| Stats mismatch | Run verification script, check for errors |

---

## Timeline

- **Preparation:** 15 minutes
- **Deployment:** 10 minutes
- **Testing:** 15 minutes
- **Production:** 30 minutes
- **Total:** ~70 minutes

---

## Notes

- Old questions in flat structure remain accessible
- New structure is backward compatible
- Cloud Functions are idempotent (safe to retry)
- Statistics update atomically (no race conditions)
- Free tier covers typical usage

---

**Deployment Date:** _______________  
**Deployed By:** _______________  
**Status:** ⬜ Not Started | ⬜ In Progress | ⬜ Complete | ⬜ Rolled Back  
**Issues Encountered:** _______________________________________________
