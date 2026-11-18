# How to run the CS Department Prolog Assistant (Windows)

## 1. One-time setup
Open PowerShell inside the project folder (the one that contains `app.py`) and run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs:
- **Flask**: the web server (`ModuleNotFoundError: No module named 'flask'` means this is missing)
- **PyMuPDF**: reads the PDFs in `AI_Knowledge_base` (imported in the code as `fitz` / `pymupdf`)

> Do **not** run `pip install fitz`. That is a different, unrelated package and breaks PyMuPDF.
> If you installed it by mistake: `python -m pip uninstall fitz frontend -y` then
> `python -m pip install --force-reinstall pymupdf`.

### Optional: use a virtual environment (recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r requirements.txt
```

## 2. Start the app
```powershell
python .\app.py
```
Then open **http://127.0.0.1:5000** in your browser. Stop the server with `Ctrl + C`.

## 3. What changed (answer precision)
The knowledge base (`AI_Knowledge_base/*.pdf`, `department_kb.pl`, `prolog_kb.pkl`) is unchanged.
A new file, `precise_answers.py`, narrows every answer to exactly what was asked:

| Question | Answer now |
|---|---|
| contact details of pranab muhuri | only his email + room |
| email of amit | the emails of both faculty named Amit |
| eligibility criteria for mca | only the MCA eligibility rule |
| total seats in mca | `Total Seats: 30` |
| duration of mca entrance test | `The duration of the Entrance Test will be 2 hours.` |
| mca fees / mca fees in inr | `USD 600 per semester` (≈ INR 54,600) |
| mca semester 2 subjects | only the semester-2 course list |
| who is the dean / list of assistant professors | only those people |
| hostel contact / hostel fees | only the hostel contact / fee line |

If something is not in the knowledge base, the app says so instead of returning an unrelated section.

## 4. Short forms the assistant understands
`maths`/`math` → mathematics, `m&c`/`mnc` → mathematics and computing, `dbms` → database systems,
`os` → operating systems, `cn` → computer networks, `ai`/`ml`/`dl`/`nlp`, `daa`, `dsa`, `oop`,
`sem` → semester, `subj` → subjects, `hod` → head of department, `dept` → department,
`cutoff` → minimum marks, `intake` → seats, `email id` → email, `ph no` → phone number, `prof`, `asst prof` …
The full list is `TERM_MAP` at the top of `precise_answers.py`; add your own lines there.

## 5. Background logo
`static/sau_logo_transparent.png` (upscaled, transparent) is shown as a faint watermark behind the chat.
Change its strength with `opacity` in the `.bg-logo` rule in `templates/index.html`.
