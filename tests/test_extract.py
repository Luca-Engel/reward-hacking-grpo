from rhg.env.extract import extract_code


def test_multiple_python_fences_take_the_last():
    text = "First try:\n```python\nx = 1\n```\nBetter:\n```python\nx = 2\n```\nDone."
    r = extract_code(text)
    assert (r.code, r.how, r.truncated_fence) == ("x = 2", "python_fence", False)


def test_python_fence_beats_later_generic_fence():
    text = "```python\nx = 1\n```\noutput:\n```\n42\n```\n"
    r = extract_code(text)
    assert r.code == "x = 1" and r.how == "python_fence"


def test_unlabeled_fence_is_used_when_no_python_fence():
    text = "Here:\n```\ndef f():\n    return 1\n```\n"
    r = extract_code(text)
    assert r.code == "def f():\n    return 1" and r.how == "generic_fence" and not r.truncated_fence


def test_last_of_several_unlabeled_fences():
    r = extract_code("```\na = 1\n```\n```\na = 2\n```")
    assert r.code == "a = 2"


def test_no_fence_gives_none():
    r = extract_code("def f(): return 1")
    assert (r.code, r.how, r.truncated_fence) == (None, "none", False)
    assert extract_code("").code is None


def test_unterminated_final_fence_counts_and_is_flagged():
    r = extract_code("Answer:\n```python\ndef f():\n    return 1\n")
    assert r.code == "def f():\n    return 1\n" and r.how == "python_fence" and r.truncated_fence is True


def test_unterminated_last_fence_wins_over_earlier_terminated_one():
    r = extract_code("```python\nx = 1\n```\nmore\n```python\nx = 2\ny = 3")
    assert r.code == "x = 2\ny = 3" and r.truncated_fence is True


def test_unterminated_generic_fence_is_lenient_too():
    r = extract_code("```\nx = 1")
    assert r.code == "x = 1" and r.how == "generic_fence" and r.truncated_fence is True


def test_python_fence_inside_prose_and_inline_backticks_ignored():
    text = (
        "I will use `sorted()` here, and ```not a fence``` inline.\n\n"
        "Some prose first.\n```python\nimport math\nprint(math.pi)\n```\nAnd some prose after."
    )
    r = extract_code(text)
    assert r.code == "import math\nprint(math.pi)" and r.how == "python_fence"


def test_other_language_fence_is_ignored_and_case_insensitive_tag():
    assert extract_code("```bash\nls\n```").code is None
    assert extract_code("```Python\nx = 1\n```").code == "x = 1"
    assert extract_code("```py\nx = 1\n```").how == "python_fence"


def test_longer_fence_can_contain_shorter_fence_lines():
    text = "````python\ns = '''\n```\n'''\n````\n"
    assert extract_code(text).code == "s = '''\n```\n'''"


def test_empty_blocks_are_not_code():
    assert extract_code("```python\n\n```").code is None
    r = extract_code("```python\nx = 1\n```\n```python\n```")
    assert r.code == "x = 1"


def test_crlf_and_indented_fence():
    r = extract_code("Text\r\n  ```python\r\nx = 1\r\n  ```\r\n")
    assert r.code == "x = 1"
