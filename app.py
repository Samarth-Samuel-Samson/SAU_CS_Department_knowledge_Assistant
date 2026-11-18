from flask import Flask, render_template, request, jsonify
from prolog_engine import PrologInferenceEngine

app = Flask(__name__)

# Compiles PDFs into 'department_kb.pl' + 'prolog_kb.pkl' and initializes the Prolog Engine
rag = PrologInferenceEngine(
    pdf_dir="AI_Knowledge_base",
    pl_file="department_kb.pl",
    pkl_file="prolog_kb.pkl"
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ask", methods=["POST"])
def ask_question():
    payload = request.get_json() or {}
    question = payload.get("question", "").strip()
    response = rag.run_rag(question)
    return jsonify(response)

@app.route("/api/reindex", methods=["POST"])
def rebuild_index():
    rag.load_or_compile(force_rebuild=True)
    total_secs = len(rag.kb["sections"]) if rag.kb else 0
    total_fac = len(rag.kb["faculty"]) if rag.kb else 0
    return jsonify({
        "status": "ok",
        "chunks": total_secs,
        "message": f"Compiled {total_secs} Prolog section facts & {total_fac} faculty facts into department_kb.pl!"
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)