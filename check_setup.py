"""
Run this from the project folder:   python .\\check_setup.py
It shows exactly which code/files are being used and tests a few questions.
"""
import os
import sys
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

print("Python       :", sys.version.split()[0], "->", sys.executable)
print("Project dir  :", HERE)

for name in ["app.py", "prolog_engine.py", "precise_answers.py", "prolog_compiler.py",
             "department_kb.pl", "prolog_kb.pkl", "templates/index.html"]:
    path = os.path.join(HERE, name)
    if os.path.exists(path):
        md5 = hashlib.md5(open(path, "rb").read()).hexdigest()[:10]
        print(f"  [ok]      {name:22s} {os.path.getsize(path):>9,} bytes  md5:{md5}")
    else:
        print(f"  [MISSING] {name}")

pdf_dir = os.path.join(HERE, "AI_Knowledge_base")
pdfs = [f for _, _, fs in os.walk(pdf_dir) for f in fs if f.lower().endswith(".pdf")] if os.path.isdir(pdf_dir) else []
print(f"PDF folder   : {pdf_dir} -> {len(pdfs)} PDFs")

try:
    import flask
    from importlib.metadata import version
    print("Flask        :", version("flask"))
except ImportError:
    print("Flask        : NOT INSTALLED  ->  python -m pip install flask")
try:
    import pymupdf
    print("PyMuPDF      :", pymupdf.VersionBind)
except ImportError:
    try:
        import fitz
        print("PyMuPDF      : (fitz)", getattr(fitz, "VersionBind", "?"))
    except ImportError:
        print("PyMuPDF      : NOT INSTALLED  ->  python -m pip install pymupdf")

import prolog_engine
print("Engine file  :", prolog_engine.__file__)
if "_run_rag_inner" not in open(prolog_engine.__file__, encoding="utf-8").read():
    print("\n!!! prolog_engine.py is the OLD version. Replace it with the one from the new zip. !!!\n")
    sys.exit(1)
import precise_answers
print("Precision    :", precise_answers.__file__, "version", precise_answers.PRECISION_VERSION)

engine = prolog_engine.PrologInferenceEngine(
    pdf_dir="AI_Knowledge_base", pl_file="department_kb.pl", pkl_file="prolog_kb.pkl")
for q in ["what is the contact details of pranab muhuri",
          "what is the eligibilty of the mca course",
          "mca fees"]:
    print("\nQ:", q)
    print(engine.run_rag(q)["answer"][:400])
