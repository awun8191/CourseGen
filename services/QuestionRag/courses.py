import logging
from typing import List
import json


class CourseModel:

    def __init__(self):
        self.semester = ""
        self.level = ""
        self.title = ""
        self.code = ""
        self.units = 0
        self.type = ""
        self.is_elective = ""

class DepartmentModel:

    def __init__(self):
        self.names: List[str] = []
        self.courses: List[CourseModel] = []





class DataFormatting:

    def __init__(self):
        self.json_document = self._read_json()
        self.courses: List[DepartmentModel] = []
        self.map_data()

        logging.info("Course Data Initialized from Json Document")
        # print(self.courses[50].courses[0])


    def _read_json(self):
        with open('/home/raregazetto/Documents/Recursive-PDF-EXTRACTION-AND-RAG/COURSEGEN/data/textbooks/courses.json') as courses_file:
            data = json.load(courses_file)

        return data

    def map_data(self):

        for value in self.json_document:
            dep = DepartmentModel()

            dep.names = value["offered_by_programs"]

            course = CourseModel()
            course.title = value["title"]
            course.semester = value["semesters"][0]
            course.type = value["type"]
            course.level = value["levels"][0]
            course.is_elective = value["is_elective"]
            course.units = value["units"]
            course.code = value["code"]

            dep.courses.append(course)
            self.courses.append(dep)
    
    def search_course(self, course_code: str):
        for i in self.courses:
            # print("="*60)
            # print(f"Department: {i.names}")
            for item in i.courses:
                if item.code.lower() == course_code.lower():
                    # COURSE_MODEL, DEPARTMENT
                    return item, i.names
                # print(f"{item.title}")
                # print(item.code)
                # print(item.level)
            # print("="*60, end="\n")
                




data = DataFormatting()

print(len(data.courses))
# for i in data.courses:
#     print("="*60)
#     print(f"Department: {i.names}")
#     for item in i.courses:
#         print(f"{item.title}")
#         print(item.code)
#         print(item.level)
#     print("="*60, end="\n")




print(data.search_course("EEE 313")[0].title)