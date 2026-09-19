from scripts.prepare_data import clean_text, format_gsm8k, format_numina, SYSTEM_PROMPT


def test_clean_text():
    assert clean_text("  hello world  \r\n") == "hello world"
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_format_gsm8k_valid():
    example = {
        "question": "Janet has 3 apples. She buys 2 more. How many?",
        "answer": "She starts with 3. <<3+2=5>> 3+2=5. #### 5",
    }
    res = format_gsm8k(example)
    assert res is not None
    assert "messages" in res
    assert len(res["messages"]) == 3
    assert res["messages"][0]["role"] == "system"
    assert res["messages"][0]["content"] == SYSTEM_PROMPT
    assert res["messages"][1]["role"] == "user"
    assert "Janet has 3 apples" in res["messages"][1]["content"]
    assert res["messages"][2]["role"] == "assistant"
    assert "<thought>" in res["messages"][2]["content"]
    assert "Final Answer: 5" in res["messages"][2]["content"]
    # Check that <<...>> calculator callout was cleaned
    assert "<<3+2=5>>" not in res["messages"][2]["content"]


def test_format_gsm8k_empty():
    assert format_gsm8k({"question": "", "answer": "foo"}) is None
    assert format_gsm8k({"question": "foo", "answer": ""}) is None


def test_format_numina_boxed():
    example = {
        "problem": "Solve for x: 2x = 6.",
        "solution": "We divide both sides by 2: x = 3. Therefore, \\boxed{3} is the answer.",
    }
    res = format_numina(example)
    assert res is not None
    assert "messages" in res
    assert "Final Answer: 3" in res["messages"][2]["content"]


def test_format_numina_length_filtering():
    # Too short
    assert format_numina({"problem": "2+2?", "solution": "4"}) is None
    # Too long
    assert format_numina({"problem": "hard problem", "solution": "A" * 3000}) is None
