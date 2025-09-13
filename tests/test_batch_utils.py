import os
import json
import tempfile

from services.QuestionRag.batch_utils import (
    slugify,
    sha256_text,
    validate_options,
    validate_answer_in_options,
    validate_calc_steps,
    write_jsonl,
)


def test_slugify_basic():
    assert slugify("BJT Amplifiers") == "bjt-amplifiers"
    assert slugify("  RF & Microwaves  ") == "rf-microwaves"


def test_sha256_text_stable():
    a = sha256_text("hello")
    b = sha256_text("hello")
    assert a == b


def test_validate_options_and_answer():
    opts = ["A", "B", "C", "D"]
    validate_options(opts)
    validate_answer_in_options("C", opts)

    try:
        validate_options(["A", "B"])  # too few
        assert False, "should have raised"
    except ValueError:
        pass

    try:
        validate_answer_in_options("E", opts)
        assert False, "should have raised"
    except ValueError:
        pass


def test_validate_calc_steps():
    ok = {"1": "step", "2": "step", "3": "step"}
    validate_calc_steps(ok)
    try:
        validate_calc_steps({"1": "only two", "2": "steps"})
        assert False, "should have raised"
    except ValueError:
        pass


def test_write_jsonl_roundtrip():
    records = [
        {"id": 1, "data": "x"},
        {"id": 2, "data": "y"},
    ]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "out.jsonl")
        write_jsonl(path, records)
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(ln) for ln in f if ln.strip()]
        assert len(lines) == 2
        assert lines[0]["id"] == 1 and lines[1]["id"] == 2

