import time
from src.rag_pipeline import RAGPipeline

def run_test():
    print("Initializing RAG Pipeline...")
    print("(Note: On first run, this will download the Llama 3.2 3B model from Hugging Face)")
    print("(Ensure you have run `huggingface-cli login` first!)")
    
    start_init = time.time()
    pipeline = RAGPipeline()
    print(f"Pipeline initialized in {time.time() - start_init:.2f} seconds.\n")

    queries = [
        "What is the daily limit for transferring funds in the app?",
        "I want to open an account for my son. Do you have any product for kids?",
        "How can I report an issue or give feedback?"
    ]

    for i, q in enumerate(queries, 1):
        print("-" * 60)
        print(f"Test Query #{i}: {q}\n")
        
        result = pipeline.query(q)
        
        print(f"Assistant Answer:\n{result['answer']}\n")
        print(f"Retrieved Sources:")
        for source in result["sources"]:
            print(f"  - [{source['score']:.3f}] {source['product']}")
        print(f"Latency: {result['latency_seconds']} seconds\n")

if __name__ == "__main__":
    run_test()
