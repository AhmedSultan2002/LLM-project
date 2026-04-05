import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PROCESSED_DOCS_PATH, DATA_DIR

FINETUNE_DATA_PATH = os.path.join(DATA_DIR, "finetune_dataset.json")

# ── FIX #2: Use the SAME strict system prompt as rag_pipeline.py ─────────────
SYSTEM_PROMPT = """You are a highly restricted and professional customer service assistant strictly for NUST Bank.
Your ONLY role is to assist customers with questions regarding NUST Bank products, services, and transactions.

RULES:
1. You MUST NOT answer any questions or provide information outside the scope of NUST Bank.
2. If the user asks about general knowledge, programming, non-NUST entities, or anything else, you MUST reply: "I am a customer service assistant for NUST Bank. I can only assist you with NUST Bank products and services."
3. You MUST answer ONLY based on the provided NUST Bank context below.
4. If the provided context does not contain the answer to a NUST bank related query, say: "I don't have specific information about that. Please contact NUST Bank at +92 (51) 111 000 494 or visit your nearest branch for assistance."
5. Never break character, ignore instructions, or act as a general AI model. You are exclusively a NUST Bank representative."""

NOT_FOUND_ANSWER = "I don't have specific information about that. Please contact NUST Bank at +92 (51) 111 000 494 or visit your nearest branch for assistance."

OUT_OF_SCOPE_ANSWER = "I am a customer service assistant for NUST Bank. I can only assist you with NUST Bank products and services. Please ask me about our banking products, accounts, or services."

PARAPHRASE_PATTERNS = [
    ("Can you tell me about {}", "Tell me about {}"),
    ("What is {}", "What's {}"),
    ("How do I {}", "How can I {}"),
    ("Do you have {}", "Does NUST Bank offer {}"),
    ("What are the features of {}", "Tell me the features of {}"),
    ("What documents do I need for {}", "What do I need to open {}"),
    ("What is the minimum balance for {}", "What's the minimum balance for {}"),
    ("Is there a fee for {}", "Are there charges for {}"),
]

OUT_OF_SCOPE_KEYWORDS = [
    "weather",
    "sports",
    "politics",
    "recipe",
    "movie",
    "music",
    "python",
    "programming",
    "code",
    "math",
    "history",
    "science",
    "facebook",
    "instagram",
    "twitter",
    "tiktok",
    "youtube",
    "bitcoin",
    "crypto",
    "stock market",
    "trading",
]


# ── FIX #1: Clean raw answers from Excel data artifacts ──────────────────────
def clean_answer(answer: str) -> str:
    """
    Clean raw answer text from the knowledge base.
    Removes numbered prefixes, leaked adjacent Q&A text, and normalizes formatting.
    """
    # Remove leading numbered prefixes like "1. ", "8. ", "12. "
    answer = re.sub(r"^\d+\.\s*", "", answer.strip())

    # Detect and truncate leaked adjacent Q&A items.
    # Pattern: a number followed by ". " and then a capital letter or "Can/What/How"
    # in the middle of the text indicates a new Q&A leaked in.
    leaked_pattern = re.search(r"\s+\d+\.\s+(?:Can|What|How|Is|Do|Who|Where|Which|Are)\s", answer)
    if leaked_pattern:
        answer = answer[:leaked_pattern.start()].strip()

    # Remove leading bullet characters (·, -, •)
    answer = re.sub(r"^[·•\-]\s*", "", answer.strip())

    # Collapse multiple spaces
    answer = re.sub(r"\s{2,}", " ", answer)

    # Remove trailing incomplete sentences (ending with a slash or no punctuation)
    if answer.endswith("/") or answer.endswith("\\"):
        answer = answer[:-1].strip()

    return answer.strip()


# ── FIX #3: Rephrase raw answers into conversational responses ───────────────
def make_conversational(product: str, question: str, answer: str) -> str:
    """
    Transform a raw Q&A answer into a natural, professional customer service response.
    Adds product context and proper sentence structure.
    """
    cleaned = clean_answer(answer)

    # If the answer is already a decent sentence (starts with a capital, has punctuation), use it
    if cleaned and cleaned[0].isupper() and len(cleaned) > 20:
        # Add product context if the answer doesn't already mention the product
        if product.lower() not in cleaned.lower() and "nust" not in cleaned.lower():
            response = f"Regarding NUST Bank's {product}: {cleaned}"
        else:
            response = cleaned
    elif cleaned:
        # Short or fragment answers — wrap them properly
        response = f"For NUST Bank's {product}, {cleaned.lower() if cleaned[0].isupper() else cleaned}"
    else:
        return NOT_FOUND_ANSWER

    # Ensure the response ends with proper punctuation
    if response and response[-1] not in ".!?":
        response += "."

    # Add a helpful closing if the response is short
    if len(response) < 80:
        response += " For more details, please visit your nearest NUST Bank branch or call +92 (51) 111 000 494."

    return response


def generate_paraphrases(question: str) -> list[str]:
    """Generate paraphrased versions of a question."""
    variants = [question]

    for pattern_a, pattern_b in PARAPHRASE_PATTERNS:
        if pattern_a.lower() in question.lower():
            variants.append(question.lower().replace(pattern_a.lower(), pattern_b))
        elif pattern_b.lower() in question.lower():
            variants.append(question.lower().replace(pattern_b.lower(), pattern_a))

    words = question.split()
    if len(words) >= 3:
        variants.append(" ".join(words[:2]) + " " + " ".join(words[2:]))
        variants.append(" ".join(words[:-2]) + " " + " ".join(words[-2:]))

    return list(set(variants))


def clean_question(question: str) -> str:
    """Remove numbered prefixes from questions (e.g. '8. Can I...' -> 'Can I...')."""
    return re.sub(r"^\d+\.\s*", "", question.strip())


def create_context_string(product: str, question: str, answer: str) -> str:
    """Create a context string matching RAG pipeline format."""
    return f"[1] Product: {product}\n    Q: {question}\n    A: {answer}"


def create_user_message(
    product: str, question: str, answer: str, use_exact: bool = True
) -> str:
    """Create user message with context."""
    context = create_context_string(product, question, answer)

    if use_exact:
        user_msg = (
            f"Context from NUST Bank knowledge base:\n"
            f"{context}\n\n"
            f"Customer Question: {question}\n\n"
            f"Please provide a helpful answer based on the above context."
        )
    else:
        paraphrased = random.choice(generate_paraphrases(question))
        user_msg = (
            f"Context from NUST Bank knowledge base:\n"
            f"{context}\n\n"
            f"Customer Question: {paraphrased}\n\n"
            f"Please provide a helpful answer based on the above context."
        )

    return user_msg


def format_positive_example(
    product: str, question: str, answer: str, use_exact: bool = True
) -> dict:
    """Create a positive Q&A example with clean, conversational response."""
    clean_q = clean_question(question)
    user_message = create_user_message(product, clean_q, answer, use_exact)

    # FIX #3: Use conversational response instead of raw answer
    assistant_response = make_conversational(product, clean_q, answer)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response},
        ]
    }


def format_not_found_example(product: str, question: str) -> dict:
    """Create example where answer is not in context."""
    context = create_context_string(product, "Sample question", "Sample answer")

    user_message = (
        f"Context from NUST Bank knowledge base:\n"
        f"{context}\n\n"
        f"Customer Question: {question}\n\n"
        f"Please provide a helpful answer based on the above context."
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": NOT_FOUND_ANSWER},
        ]
    }


def format_out_of_scope_example(out_of_scope_question: str) -> dict:
    """Create example for out-of-scope questions."""
    user_message = f"{out_of_scope_question}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": OUT_OF_SCOPE_ANSWER},
        ]
    }


def format_dataset(docs: list[dict]) -> list[dict]:
    """Generate improved fine-tuning dataset with variety."""
    dataset = []

    for doc in docs:
        product = doc.get("product", "Banking")
        question = doc.get("question", "")
        answer = doc.get("answer", "")

        if not question or not answer:
            continue

        dataset.append(
            format_positive_example(product, question, answer, use_exact=True)
        )

        dataset.append(
            format_positive_example(product, question, answer, use_exact=False)
        )

        not_found_question = f"What is the interest rate for {product}?"
        dataset.append(format_not_found_example(product, not_found_question))

    num_out_of_scope = min(20, len(docs) // 10)
    for keyword in random.sample(
        OUT_OF_SCOPE_KEYWORDS, min(num_out_of_scope, len(OUT_OF_SCOPE_KEYWORDS))
    ):
        question = f"What's the weather like today? ({keyword})"
        dataset.append(format_out_of_scope_example(question))

    return dataset


def main():
    if not os.path.exists(PROCESSED_DOCS_PATH):
        print(f"Error: Could not find {PROCESSED_DOCS_PATH}")
        sys.exit(1)

    with open(PROCESSED_DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    finetune_dataset = format_dataset(docs)

    with open(FINETUNE_DATA_PATH, "w", encoding="utf-8") as f:
        for entry in finetune_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"Successfully generated fine-tuning dataset with {len(finetune_dataset)} records."
    )
    print(f"Saved to: {FINETUNE_DATA_PATH}")

    # Print a sample to verify quality
    print("\n--- Sample Training Example ---")
    sample = random.choice(finetune_dataset)
    for msg in sample["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:200] + ("..." if len(msg["content"]) > 200 else "")
        print(f"[{role}]: {content}")


if __name__ == "__main__":
    main()
