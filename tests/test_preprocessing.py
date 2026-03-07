"""
Unit tests for data_preprocessing.py

These tests cover pure-Python logic and do NOT require a GPU,
a Hugging Face token, or any large model download.
"""

import pytest

from src.data_preprocessing import (
    clean_text,
    _is_question,
    parse_json_faq,
    build_corpus,
)


# ──────────────────────────────────────────────────────────────────────────────
# clean_text
# ──────────────────────────────────────────────────────────────────────────────


class TestCleanText:
    def test_returns_empty_string_for_none(self):
        assert clean_text(None) == ""

    def test_returns_empty_string_for_non_string(self):
        assert clean_text(123) == ""

    def test_strips_leading_and_trailing_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self):
        assert clean_text("hello   world\n\tfoo") == "hello world foo"

    def test_replaces_smart_double_quotes(self):
        assert clean_text("\u201chello\u201d") == '"hello"'

    def test_replaces_smart_single_quotes(self):
        assert clean_text("it\u2019s") == "it's"

    def test_replaces_right_arrow(self):
        assert clean_text("Step 1 \u2192 Step 2") == "Step 1 -> Step 2"

    def test_empty_string_passthrough(self):
        assert clean_text("") == ""

    def test_plain_text_unchanged(self):
        assert clean_text("Hello World") == "Hello World"


# ──────────────────────────────────────────────────────────────────────────────
# _is_question
# ──────────────────────────────────────────────────────────────────────────────


class TestIsQuestion:
    def test_ends_with_question_mark(self):
        assert _is_question("What is the interest rate?") is True

    def test_starts_with_what(self):
        assert _is_question("What documents are required") is True

    def test_starts_with_how(self):
        assert _is_question("How do I apply") is True

    def test_starts_with_can(self):
        assert _is_question("Can I transfer funds online") is True

    def test_starts_with_is(self):
        assert _is_question("Is there a minimum balance") is True

    def test_starts_with_does(self):
        assert _is_question("Does this account earn profit") is True

    def test_starts_with_will(self):
        assert _is_question("Will I get a debit card") is True

    def test_starts_with_are(self):
        assert _is_question("Are there any fees") is True

    def test_starts_with_which(self):
        assert _is_question("Which branch should I visit") is True

    def test_starts_with_who(self):
        assert _is_question("Who can open this account") is True

    def test_starts_with_when(self):
        assert _is_question("When will my card arrive") is True

    def test_starts_with_for_which(self):
        assert _is_question("For which customers is this available") is True

    def test_starts_with_in_case(self):
        assert _is_question("In case of loss, what should I do?") is True

    def test_plain_statement_is_not_question(self):
        assert _is_question("The minimum balance is Rs. 5000.") is False

    def test_empty_string_is_not_question(self):
        assert _is_question("") is False

    def test_case_insensitive_matching(self):
        assert _is_question("WHAT is the rate?") is True


# ──────────────────────────────────────────────────────────────────────────────
# parse_json_faq
# ──────────────────────────────────────────────────────────────────────────────


class TestParseJsonFaq:
    def test_returns_documents_for_valid_input(self, tmp_path):
        import json

        faq_data = {
            "categories": [
                {
                    "category": "Transfers",
                    "questions": [
                        {
                            "question": "What is the daily limit?",
                            "answer": "Rs. 500,000",
                        },
                        {
                            "question": "How do I cancel a transfer?",
                            "answer": "Go to history tab.",
                        },
                    ],
                }
            ]
        }
        faq_file = tmp_path / "faq.json"
        faq_file.write_text(json.dumps(faq_data), encoding="utf-8")

        docs = parse_json_faq(str(faq_file))

        assert len(docs) == 2
        assert docs[0]["question"] == "What is the daily limit?"
        assert docs[0]["answer"] == "Rs. 500,000"
        assert docs[0]["product"] == "Mobile App — Transfers"
        assert docs[0]["source"] == "json:funds_transfer_faq"

    def test_skips_entries_with_empty_question(self, tmp_path):
        import json

        faq_data = {
            "categories": [
                {
                    "category": "General",
                    "questions": [
                        {"question": "", "answer": "Some answer"},
                        {"question": "Valid question?", "answer": "Valid answer"},
                    ],
                }
            ]
        }
        faq_file = tmp_path / "faq.json"
        faq_file.write_text(json.dumps(faq_data), encoding="utf-8")

        docs = parse_json_faq(str(faq_file))

        assert len(docs) == 1
        assert docs[0]["question"] == "Valid question?"

    def test_skips_entries_with_empty_answer(self, tmp_path):
        import json

        faq_data = {
            "categories": [
                {
                    "category": "General",
                    "questions": [
                        {"question": "What?", "answer": ""},
                    ],
                }
            ]
        }
        faq_file = tmp_path / "faq.json"
        faq_file.write_text(json.dumps(faq_data), encoding="utf-8")

        docs = parse_json_faq(str(faq_file))

        assert len(docs) == 0

    def test_empty_categories_returns_empty_list(self, tmp_path):
        import json

        faq_file = tmp_path / "faq.json"
        faq_file.write_text(json.dumps({"categories": []}), encoding="utf-8")

        docs = parse_json_faq(str(faq_file))

        assert docs == []

    def test_cleans_unicode_in_questions_and_answers(self, tmp_path):
        import json

        faq_data = {
            "categories": [
                {
                    "category": "Info",
                    "questions": [
                        {
                            "question": "\u201cWhat is this?\u201d",
                            "answer": "It\u2019s a bank.",
                        },
                    ],
                }
            ]
        }
        faq_file = tmp_path / "faq.json"
        faq_file.write_text(json.dumps(faq_data), encoding="utf-8")

        docs = parse_json_faq(str(faq_file))

        assert docs[0]["question"] == '"What is this?"'
        assert docs[0]["answer"] == "It's a bank."


# ──────────────────────────────────────────────────────────────────────────────
# Deduplication in build_corpus
# ──────────────────────────────────────────────────────────────────────────────


class TestDeduplication:
    def test_duplicate_question_product_pairs_are_removed(self, tmp_path, monkeypatch):
        """
        Patch parse_excel and parse_json_faq to return controlled duplicates,
        then verify build_corpus deduplicates them.
        """
        import src.data_preprocessing as dp

        duplicate_doc = {
            "question": "What is the Target Market?",
            "answer": "Salaried individuals.",
            "product": "NUST Savings Account",
            "source": "excel:Sheet1",
        }
        # Same question + product, different source sheet — should be deduplicated
        duplicate_doc_2 = {
            "question": "What is the Target Market?",
            "answer": "Salaried individuals.",
            "product": "NUST Savings Account",
            "source": "excel:Sheet2",
        }

        monkeypatch.setattr(
            dp, "parse_excel", lambda path: [duplicate_doc, duplicate_doc_2]
        )
        monkeypatch.setattr(dp, "parse_json_faq", lambda path: [])
        # Ensure file existence checks pass
        monkeypatch.setattr(dp.os.path, "exists", lambda path: True)

        corpus = dp.build_corpus()

        matching = [d for d in corpus if d["question"] == "What is the Target Market?"]
        assert len(matching) == 1, "Duplicate Q+product pair should appear only once"

    def test_same_question_different_product_is_kept(self, tmp_path, monkeypatch):
        import src.data_preprocessing as dp

        doc_a = {
            "question": "What is the minimum balance?",
            "answer": "Rs. 5000",
            "product": "NUST Savings Account",
            "source": "excel:Sheet1",
        }
        doc_b = {
            "question": "What is the minimum balance?",
            "answer": "Rs. 10000",
            "product": "NUST Current Account",
            "source": "excel:Sheet2",
        }

        monkeypatch.setattr(dp, "parse_excel", lambda path: [doc_a, doc_b])
        monkeypatch.setattr(dp, "parse_json_faq", lambda path: [])
        monkeypatch.setattr(dp.os.path, "exists", lambda path: True)

        corpus = dp.build_corpus()

        assert len(corpus) == 2, (
            "Same question for different products must both be retained"
        )
