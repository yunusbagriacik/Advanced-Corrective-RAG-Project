from dotenv import load_dotenv

load_dotenv()

from graph.graph import app #graph.py dosyası çalışır, graph kurulur ve compile edilir.

if __name__ == "__main__":
    print(app.invoke(input={"question": "What is adverserial attack on LLM?"}))