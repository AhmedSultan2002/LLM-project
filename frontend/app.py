"""
NUST Bank Customer Service Assistant - Frontend
================================================
Streamlit web interface for the RAG-powered chatbot.

Usage:
    streamlit run frontend/app.py
"""

import streamlit as st
import httpx
import time
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000/api/v1"
API_TIMEOUT = 120  # seconds (LLM can be slow to load)

# ─── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NUST Bank Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stChat {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
    }
    .source-card {
        background-color: #fff3e0;
        padding: 10px 14px;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 6px 0;
        font-size: 0.9em;
    }
    .metric-card {
        background-color: #e8f5e9;
        padding: 8px 12px;
        border-radius: 6px;
        display: inline-block;
        margin: 4px;
    }
    .stButton>button {
        background-color: #1976d2;
        color: white;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
    .loading-spinner {
        text-align: center;
        padding: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Session State ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_connected" not in st.session_state:
    st.session_state.api_connected = False


# ─── API Client ─────────────────────────────────────────────────────────────
class APIClient:
    """Simple wrapper around the FastAPI endpoints."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=API_TIMEOUT)

    def check_health(self) -> dict:
        """Check if the API is running."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def submit_query(self, query: str) -> dict:
        """Submit a query to the RAG pipeline."""
        try:
            response = self.client.post(
                f"{self.base_url}/query",
                json={"query": query},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def get_sources(self) -> list:
        """Get available document sources."""
        try:
            response = self.client.get(f"{self.base_url}/sources")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return []

    def add_document(self, product: str, question: str, answer: str) -> dict:
        """Add a new document to the knowledge base."""
        try:
            response = self.client.post(
                f"{self.base_url}/documents",
                json={"product": product, "question": question, "answer": answer},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"success": False, "message": f"HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"success": False, "message": str(e)}


# ─── UI Components ───────────────────────────────────────────────────────────
def show_header():
    """Display the app header."""
    col1, col2 = st.columns([6, 2])
    with col1:
        st.title("🏦 NUST Bank Customer Service Assistant")
        st.caption("Powered by Llama 3.2 with RAG | Fine-tuned with QLoRA")
    with col2:
        st.markdown("###")
        if st.session_state.api_connected:
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Offline")


def show_sidebar():
    """Display sidebar with information and controls."""
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            """
            **NUST Bank Assistant** helps customers with questions about:
            - Bank products and services
            - Account information
            - Funds transfer procedures
            - And more...
            """
        )

        st.divider()

        st.header("⚙️ Settings")
        api_url = st.text_input(
            "API Base URL",
            value=API_BASE_URL,
            help="URL of the FastAPI backend server",
        )

        if st.button("🔄 Reconnect to API"):
            st.session_state.api_connected = False
            client = APIClient(api_url)
            health = client.check_health()
            st.session_state.api_connected = health.get("status") == "ok"
            st.rerun()

        st.divider()

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        st.header("📄 Add New Document")
        st.caption("Add a new Q&A entry to the live knowledge base.")
        with st.form("add_doc_form", clear_on_submit=True):
            new_product = st.text_input("Product / Category", placeholder="e.g. Home Loan")
            new_question = st.text_area("Question", placeholder="e.g. What is the maximum loan tenure?", height=80)
            new_answer = st.text_area("Answer", placeholder="The maximum tenure is 20 years...", height=120)
            submitted = st.form_submit_button("➕ Add to Knowledge Base")

        if submitted:
            if not new_product or not new_question or not new_answer:
                st.error("All three fields are required.")
            else:
                client = APIClient()
                result = client.add_document(new_product, new_question, new_answer)
                if result.get("success"):
                    st.success(result.get("message", "Document added."))
                else:
                    st.error(result.get("message", "Failed to add document."))

        st.divider()

        st.header("📊 Session Stats")
        st.markdown(
            f"**Messages:** {len(st.session_state.messages)}",
        )


def show_chat():
    """Display the chat interface."""
    st.subheader("💬 Chat")

    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show sources for assistant messages
            if msg["role"] == "assistant" and "sources" in msg:
                if msg["sources"]:
                    with st.expander("📚 View Sources"):
                        for source in msg["sources"]:
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <strong>{source.get("product", "N/A")}</strong>
                                    <br>{source.get("question", "")}
                                    <br><em>Relevance: {source.get("score", 0):.3f}</em>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            # Show latency for assistant messages
            if msg["role"] == "assistant" and "latency" in msg:
                st.caption(f"⏱️ Response time: {msg['latency']}s")

    # Chat input
    if prompt := st.chat_input("Ask a question about NUST Bank services..."):
        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "timestamp": datetime.now()}
        )

        # Get API client
        client = APIClient()

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                start_time = time.time()
                response = client.submit_query(prompt)
                elapsed = time.time() - start_time

            if "error" in response:
                st.error(f"Error: {response['error']}")
                result_content = "Sorry, I encountered an error. Please try again."
            else:
                st.markdown(response.get("answer", "No response generated."))
                result_content = response.get("answer", "")

                # Show sources if available
                if response.get("sources"):
                    with st.expander("📚 View Sources"):
                        for source in response["sources"]:
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <strong>{source.get("product", "N/A")}</strong>
                                    <br>{source.get("question", "")}
                                    <br><em>Relevance: {source.get("score", 0):.3f}</em>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.caption(f"⏱️ Response time: {elapsed:.2f}s")

        # Add assistant response to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result_content,
                "sources": response.get("sources", []),
                "latency": round(elapsed, 2),
                "timestamp": datetime.now(),
            }
        )


def check_api_connection():
    """Check API connection on startup."""
    if not st.session_state.api_connected:
        client = APIClient()
        health = client.check_health()
        st.session_state.api_connected = health.get("status") == "ok"


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    check_api_connection()
    show_header()
    show_sidebar()
    show_chat()


if __name__ == "__main__":
    main()
