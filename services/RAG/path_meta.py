import re
from pathlib import Path
from typing import Dict

def _clean_level(s: str) -> str:
    if not s:
        return ""
    s2 = re.sub(r"(?i)level", "", s)
    m = re.search(r"(100|200|300|400|500)", s2)
    return m.group(1) if m else ""

SEM = {
    "1": "1", "2": "2", "FIRST": "1", "SECOND": "2",
    "SEM1": "1", "SEM2": "2", "SEMESTER1": "1", "SEMESTER2": "2",
}


def parse_path_meta(path: Path) -> Dict[str, str]:
    parts = path.parts
    filename = parts[-1] if len(parts) >= 1 else ""
    course_folder = parts[-2] if len(parts) >= 2 else ""
    semester_raw = parts[-3] if len(parts) >= 3 else ""
    level_raw = parts[-4] if len(parts) >= 4 else ""
    dept = parts[-5] if len(parts) >= 5 else ""

    level = _clean_level(level_raw)
    sem = SEM.get(semester_raw.strip().upper().replace(" ", ""), "")

    m = re.search(r"([A-Za-z]{2,})\s*[-_ ]*\s*(\d{2,3})", course_folder)
    code, num = (m.group(1).upper(), m.group(2)) if m else ("", "")
    if not code or not num:
        m2 = re.search(r"([A-Za-z]{2,})\s*[-_ ]*\s*(\d{2,3})", Path(filename).stem)
        if m2:
            code, num = m2.group(1).upper(), m2.group(2)

    if not level and num and len(num) >= 3:
        level = num[0] + "00" if num[0] in "12345" else ""

    cf_up = course_folder.upper()
    fn_up = filename.upper()
    cat = (
        "PQ" if (cf_up in {"PQ", "PQS", "PASTQUESTIONS"} or "PQ" in fn_up or "PAST QUESTION" in fn_up or "PAST QUESTIONS" in fn_up)
        else ("GENERAL" if cf_up == "GENERAL" else "")
    )

    group_key = (
        f"{dept}-{code}-{num}" if (dept and code and num)
        else f"{code}-{num}" if (code and num)
        else dept or code or "MISC"
    )

    return {
        "DEPARTMENT": dept, "LEVEL": level, "SEMESTER": sem, "CATEGORY": cat,
        "COURSE_FOLDER": course_folder, "COURSE_CODE": code, "COURSE_NUMBER": num,
        "FILENAME": filename, "STEM": Path(filename).stem, "GROUP_KEY": group_key,
    }

