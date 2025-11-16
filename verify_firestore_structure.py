#!/usr/bin/env python3
"""Verify new Firestore structure and statistics accuracy."""

from services.Firestore.firebase_service import FireStore
from collections import defaultdict


def verify_course_stats(course_code: str) -> dict:
    """Verify statistics for a specific course."""
    store = FireStore()
    
    if not store.db:
        print("❌ Firestore not initialized")
        return {}
    
    print(f"\n📊 Verifying statistics for: {course_code}")
    print("=" * 60)
    
    # Get stats document
    stats_ref = store.db.collection("Questions").document(course_code)
    stats_doc = stats_ref.get()
    
    if not stats_doc.exists:
        print(f"⚠️  No stats document found for {course_code}")
        return {}
    
    stats = stats_doc.to_dict()
    
    # Count actual questions
    questions_ref = stats_ref.collection("questions")
    questions = list(questions_ref.stream())
    
    actual_counts = {
        "total": len(questions),
        "theory": 0,
        "calculation": 0,
        "difficulty": defaultdict(int),
        "type_difficulty": {
            "theory": defaultdict(int),
            "calculation": defaultdict(int)
        }
    }
    
    for q_doc in questions:
        q_data = q_doc.to_dict()
        q_type = q_data.get("question_type")
        difficulty = q_data.get("difficulty")
        
        if q_type == "theory":
            actual_counts["theory"] += 1
        elif q_type == "calculation":
            actual_counts["calculation"] += 1
        
        if difficulty:
            actual_counts["difficulty"][difficulty] += 1
            if q_type in actual_counts["type_difficulty"]:
                actual_counts["type_difficulty"][q_type][difficulty] += 1
    
    # Compare stats vs actual
    print("\n📈 Statistics Comparison:")
    print(f"  Total Questions:")
    print(f"    Stats:  {stats.get('total_questions', 0)}")
    print(f"    Actual: {actual_counts['total']}")
    print(f"    ✅ Match" if stats.get('total_questions') == actual_counts['total'] else "    ❌ Mismatch")
    
    print(f"\n  Theory Questions:")
    print(f"    Stats:  {stats.get('theory_questions', 0)}")
    print(f"    Actual: {actual_counts['theory']}")
    print(f"    ✅ Match" if stats.get('theory_questions') == actual_counts['theory'] else "    ❌ Mismatch")
    
    print(f"\n  Calculation Questions:")
    print(f"    Stats:  {stats.get('calculation_questions', 0)}")
    print(f"    Actual: {actual_counts['calculation']}")
    print(f"    ✅ Match" if stats.get('calculation_questions') == actual_counts['calculation'] else "    ❌ Mismatch")
    
    print(f"\n  Difficulty Breakdown:")
    for difficulty in ["Easy", "Medium", "Hard"]:
        stats_count = stats.get('difficulty_breakdown', {}).get(difficulty, 0)
        actual_count = actual_counts['difficulty'][difficulty]
        match = "✅" if stats_count == actual_count else "❌"
        print(f"    {difficulty:8s}: Stats={stats_count:3d}, Actual={actual_count:3d} {match}")
    
    print(f"\n  Type × Difficulty Breakdown:")
    for q_type in ["theory", "calculation"]:
        print(f"    {q_type.capitalize()}:")
        for difficulty in ["Easy", "Medium", "Hard"]:
            stats_count = stats.get('type_difficulty_breakdown', {}).get(q_type, {}).get(difficulty, 0)
            actual_count = actual_counts['type_difficulty'][q_type][difficulty]
            match = "✅" if stats_count == actual_count else "❌"
            print(f"      {difficulty:8s}: Stats={stats_count:3d}, Actual={actual_count:3d} {match}")
    
    return {
        "stats": stats,
        "actual": actual_counts,
        "match": stats.get('total_questions') == actual_counts['total']
    }


def list_all_courses() -> list:
    """List all courses with questions."""
    store = FireStore()
    
    if not store.db:
        print("❌ Firestore not initialized")
        return []
    
    print("\n📚 Courses with Questions:")
    print("=" * 60)
    
    courses_ref = store.db.collection("Questions")
    courses = courses_ref.stream()
    
    course_list = []
    for course_doc in courses:
        course_code = course_doc.id
        stats = course_doc.to_dict()
        
        # Skip if it's an old-style question document (has 'question' field)
        if 'question' in stats:
            continue
        
        total = stats.get('total_questions', 0)
        theory = stats.get('theory_questions', 0)
        calc = stats.get('calculation_questions', 0)
        
        print(f"  {course_code:15s} | Total: {total:4d} | Theory: {theory:4d} | Calc: {calc:4d}")
        course_list.append(course_code)
    
    return course_list


def main():
    """Main verification routine."""
    print("🔍 Firestore Structure Verification")
    print("=" * 60)
    
    # List all courses
    courses = list_all_courses()
    
    if not courses:
        print("\n⚠️  No courses found with new structure")
        print("   Generate questions to populate the database")
        return
    
    # Verify each course
    print(f"\n\n🔬 Detailed Verification")
    print("=" * 60)
    
    all_match = True
    for course_code in courses:
        result = verify_course_stats(course_code)
        if result and not result.get('match'):
            all_match = False
    
    print("\n" + "=" * 60)
    if all_match:
        print("✅ All statistics verified successfully!")
    else:
        print("⚠️  Some statistics mismatches found")
        print("   This may indicate Cloud Functions are not deployed or not triggering")
        print("   Run: cd firebase_functions && ./deploy.sh")


if __name__ == "__main__":
    main()
