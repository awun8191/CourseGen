import json
from typing import Dict, List, Any
import pprint

def courses():
    with open("/home/raregazetto/Documents/Recursive-PDF-EXTRACTION-AND-RAG/data/textbooks/courses.json") as courses_json:
        data = json.load(courses_json)
    
    filtered = []
    for course in data:
        for key, valu in course.items():
            if key == "offered_by_programs":
                if 'AERONAUTICAL ENGINEERING' in valu:
                    filtered.append(course)
            
    # pprint.pprint(filtered)
        pprint.pprint(course)




def generate_topics_and_descriptions(course: str, department: str):
    pass