/**
 * Firebase Cloud Functions for CourseGen Question Statistics
 * 
 * Tracks question counts and statistics when questions are added to Firestore.
 * Structure: Questions/{course_code}/questions/{question_id}
 */

const functions = require('firebase-functions');
const admin = require('firebase-admin');

admin.initializeApp();
const db = admin.firestore();

/**
 * Triggered when a question is added to Questions/{course_code}/questions subcollection
 * Updates the course document with aggregated statistics
 */
exports.updateQuestionStats = functions.firestore
  .document('Questions/{courseCode}/questions/{questionId}')
  .onCreate(async (snap, context) => {
    const courseCode = context.params.courseCode;
    const questionData = snap.data();
    
    try {
      const courseDocRef = db.collection('Questions').doc(courseCode);
      
      // Use transaction to ensure atomic updates
      await db.runTransaction(async (transaction) => {
        const courseDoc = await transaction.get(courseDocRef);
        
        // Initialize stats if document doesn't exist
        let stats = {
          course_code: courseCode,
          course_name: questionData.course_name || '',
          total_questions: 0,
          theory_questions: 0,
          calculation_questions: 0,
          difficulty_breakdown: {
            Easy: 0,
            Medium: 0,
            Hard: 0
          },
          type_difficulty_breakdown: {
            theory: { Easy: 0, Medium: 0, Hard: 0 },
            calculation: { Easy: 0, Medium: 0, Hard: 0 }
          },
          last_updated: admin.firestore.FieldValue.serverTimestamp(),
          created_at: admin.firestore.FieldValue.serverTimestamp()
        };
        
        // If document exists, get current stats
        if (courseDoc.exists) {
          const existingData = courseDoc.data();
          stats = {
            ...stats,
            ...existingData,
            created_at: existingData.created_at || stats.created_at
          };
        }
        
        // Increment counters
        stats.total_questions += 1;
        
        // Update question type counts
        const questionType = questionData.question_type;
        if (questionType === 'theory') {
          stats.theory_questions += 1;
        } else if (questionType === 'calculation') {
          stats.calculation_questions += 1;
        }
        
        // Update difficulty breakdown
        const difficulty = questionData.difficulty;
        if (difficulty && stats.difficulty_breakdown[difficulty] !== undefined) {
          stats.difficulty_breakdown[difficulty] += 1;
        }
        
        // Update type-specific difficulty breakdown
        if (questionType && difficulty && 
            stats.type_difficulty_breakdown[questionType] &&
            stats.type_difficulty_breakdown[questionType][difficulty] !== undefined) {
          stats.type_difficulty_breakdown[questionType][difficulty] += 1;
        }
        
        // Update timestamp
        stats.last_updated = admin.firestore.FieldValue.serverTimestamp();
        
        // Write updated stats
        transaction.set(courseDocRef, stats, { merge: true });
      });
      
      console.log(`Successfully updated stats for course: ${courseCode}`);
      return null;
    } catch (error) {
      console.error(`Error updating stats for course ${courseCode}:`, error);
      throw error;
    }
  });

/**
 * Triggered when a question is deleted from Questions/{course_code}/questions subcollection
 * Decrements the course statistics
 */
exports.decrementQuestionStats = functions.firestore
  .document('Questions/{courseCode}/questions/{questionId}')
  .onDelete(async (snap, context) => {
    const courseCode = context.params.courseCode;
    const questionData = snap.data();
    
    try {
      const courseDocRef = db.collection('Questions').doc(courseCode);
      
      await db.runTransaction(async (transaction) => {
        const courseDoc = await transaction.get(courseDocRef);
        
        if (!courseDoc.exists) {
          console.warn(`Course document ${courseCode} not found for decrement`);
          return;
        }
        
        const stats = courseDoc.data();
        
        // Decrement counters (ensure they don't go below 0)
        stats.total_questions = Math.max(0, (stats.total_questions || 0) - 1);
        
        const questionType = questionData.question_type;
        if (questionType === 'theory') {
          stats.theory_questions = Math.max(0, (stats.theory_questions || 0) - 1);
        } else if (questionType === 'calculation') {
          stats.calculation_questions = Math.max(0, (stats.calculation_questions || 0) - 1);
        }
        
        // Decrement difficulty breakdown
        const difficulty = questionData.difficulty;
        if (difficulty && stats.difficulty_breakdown && stats.difficulty_breakdown[difficulty] !== undefined) {
          stats.difficulty_breakdown[difficulty] = Math.max(0, stats.difficulty_breakdown[difficulty] - 1);
        }
        
        // Decrement type-specific difficulty breakdown
        if (questionType && difficulty && 
            stats.type_difficulty_breakdown &&
            stats.type_difficulty_breakdown[questionType] &&
            stats.type_difficulty_breakdown[questionType][difficulty] !== undefined) {
          stats.type_difficulty_breakdown[questionType][difficulty] = 
            Math.max(0, stats.type_difficulty_breakdown[questionType][difficulty] - 1);
        }
        
        stats.last_updated = admin.firestore.FieldValue.serverTimestamp();
        
        transaction.set(courseDocRef, stats, { merge: true });
      });
      
      console.log(`Successfully decremented stats for course: ${courseCode}`);
      return null;
    } catch (error) {
      console.error(`Error decrementing stats for course ${courseCode}:`, error);
      throw error;
    }
  });
