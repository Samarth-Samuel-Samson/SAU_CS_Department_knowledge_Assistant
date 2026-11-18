"""
precise_answers.py
------------------
Precision layer for the Prolog inference engine.

The knowledge base (PDFs, department_kb.pl, prolog_kb.pkl) is NOT modified.
This module only *reads* it and narrows every answer down to the exact part
the user asked for:

  * Faculty queries   -> only the asked attribute(s) of only that person
                         (email / room / designation / qualification ...)
  * Department office -> only the CSE office contact block
  * Fee queries       -> only the fee row of the asked programme
  * Semester queries  -> only the courses of the asked semester
  * Fact queries      -> only the line(s) that contain the answer
                         ("total seats", "duration of entrance test", ...)
  * Other queries     -> only the best matching sub-section of a section
                         (e.g. "Eligibility" instead of Eligibility + Modes of
                          Admission + Seat Matrix + Test Format)
"""

import os
import re

PRECISION_VERSION = "2.3"
import math
import difflib
from collections import Counter

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # older PyMuPDF
    import fitz


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
WORD_RE = re.compile(r"[a-z0-9]+")

GENERIC_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "for", "in", "on", "at", "to",
    "and", "or", "by", "with", "from", "as", "it", "its", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "how", "when", "where", "why", "do", "does", "did",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "his", "her", "him", "they", "them",
    "please", "tell", "give", "show", "list", "know", "want", "need", "can", "could", "would",
    "should", "will", "shall", "may", "about", "details", "detail", "information", "info",
    "regarding", "related", "provide", "get", "find", "all", "any", "some", "there", "here",
    "sau", "south", "asian", "university", "department", "dept", "cse", "cs", "computer",
    "science", "engineering", "programme", "program", "course", "courses", "degree", "student",
    "students", "kindly", "required", "require", "much", "many", "much", "also", "only",
    "exact", "exactly", "sir", "mam", "maam", "madam", "dr", "prof", "professor", "mr", "ms", "mrs",
}

PROGRAM_WORDS = {
    "mca", "msc", "mtech", "btech", "phd", "bsc", "dual", "integrated", "m", "b", "tech",
    "sc", "ph", "d", "master", "masters", "bachelor", "bachelors", "applications", "application",
    "technology", "mathematics", "computing", "mnc", "doctoral", "doctorate",
}

# Words that mean the same thing for matching purposes
SYNONYMS = {
    "eligibility": "eligib", "eligible": "eligib", "criteria": "eligib", "criterion": "eligib",
    "qualification": "eligib", "qualifications": "eligib", "qualify": "eligib",
    "requirement": "eligib", "requirements": "eligib",
    "seats": "seat", "intake": "seat", "vacancies": "seat",
    "subjects": "subject", "papers": "subject",
    "fees": "fee", "charges": "fee", "charge": "fee", "cost": "fee", "tuition": "fee",
    "exam": "test", "examination": "test", "entrance": "entrance",
    "hrs": "hour", "hours": "hour", "time": "duration", "long": "duration",
    "marks": "mark", "percent": "percentage", "%": "percentage",
    "modes": "mode", "admissions": "admission", "apply": "admission",
    "credits": "credit", "semesters": "semester", "sem": "semester",
    "years": "year", "yrs": "year",
    "questions": "question", "mcqs": "question", "mcq": "question",
    "labs": "lab", "laboratories": "lab", "laboratory": "lab",
    "electives": "elective", "tracks": "track", "specialisation": "specialization",
    "specializations": "specialization", "specialisations": "specialization",
}

ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8,
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8,
}

# A query containing one of these asks for a specific fact, not a whole section
FACT_CUE_RE = re.compile(
    r"\b(how\s+many|how\s+much|how\s+long|duration|total|number\s+of|no\.?\s+of|minimum|maximum|"
    r"percentage|percent|marks?|negative|calculators?|log\s+tables?|questions?|seats?|intake|"
    r"when|date|deadline|age\s+limit|is\s+there|are\s+there|exchange\s+rate|installments?|"
    r"refunds?|credits?\s+(?:of|for)|which\s+(?:test|exam)|hours?)\b",
    re.IGNORECASE,
)


def norm_token(tok):
    tok = tok.lower()
    if tok in SYNONYMS:
        return SYNONYMS[tok]
    if len(tok) > 4 and tok.endswith("ies"):
        tok = tok[:-3] + "y"
    elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    return SYNONYMS.get(tok, tok)


def content_tokens(text, drop_programs=True):
    """Meaningful, normalised tokens of a text (used for precise matching)."""
    out = []
    for w in WORD_RE.findall(text.lower().replace("%", " percent ")):
        if w in GENERIC_WORDS:
            continue
        if drop_programs and w in PROGRAM_WORDS:
            continue
        if len(w) < 2 and not w.isdigit():
            continue
        out.append(norm_token(w))
    return out


def pretty_email(raw):
    e = raw.strip().replace(" ", "")
    e = re.sub(r"\(at\)", "@", e, flags=re.IGNORECASE)
    return e.strip(".,;")


SHORT_WORDS = {"of", "to", "in", "on", "an", "and", "the", "for", "at", "by", "or", "as", "is",
               "it", "be", "with", "via", "per", "up", "ii", "iii", "iv", "vi", "vii", "viii"}


# ---------------------------------------------------------------------------
# Word-fragment repair ("Programm ing" -> "Programming") for display only
# ---------------------------------------------------------------------------
class TextRepair:
    def __init__(self, kb):
        counts = Counter()
        for sec in kb.get("sections", []):
            for ln in sec.get("lines", []):
                if "|" in ln or ln.startswith("##SUBSECTION##"):
                    continue
                words = ln.split()
                if len(words) >= 8:  # running prose -> words are complete
                    for w in re.findall(r"[A-Za-z]+", ln):
                        if len(w) >= 3:
                            counts[w.lower()] += 1
        self.vocab = set(counts)

    GLUE_SUFFIXES = ("using", "with", "and", "for", "of", "in")

    def _split_glued(self, word):
        """'FundamentalsofComputerProgramming' -> 'Fundamentals of Computer Programming'"""
        transitions = len(re.findall(r"[a-z][A-Z&]", word))
        if not ((len(word) >= 12 and transitions >= 1) or (len(word) >= 8 and transitions >= 2)):
            return word
        parts = re.split(r"(?<=[a-z])(?=[A-Z&])|(?<=&)(?=[A-Za-z])", word)
        out = []
        for p in parts:
            low = p.lower()
            for suf in self.GLUE_SUFFIXES:
                if low.endswith(suf) and len(p) - len(suf) >= 2 and low != suf and \
                        (suf != "in" or low[:-2] in self.vocab):
                    p = p[:-len(suf)] + " " + p[-len(suf):]
                    break
            out.append(p)
        return " ".join(out).replace("& ", "& ").replace(" &", " &")

    def fix(self, text):
        text = re.sub(r"\b(\d) (\d) \.", r"\1.\2", text)       # "4 5 ." -> "4.5"
        text = re.sub(r"(?<![\d.])(\d)(\d) \.(?=\s|$|\|)", r"\1.\2", text)  # "45 ." -> "4.5"
        text = " ".join(self._split_glued(w) for w in text.split(" "))
        parts = text.split(" ")
        out = []
        for part in parts:
            if out:
                prev = out[-1]
                m_prev = re.fullmatch(r"([A-Za-z]+)", prev)
                m_cur = re.match(r"([A-Za-z]+)(.*)$", part)
                if m_prev and m_cur:
                    a, b, tail = m_prev.group(1), m_cur.group(1), m_cur.group(2)
                    joined = (a + b).lower()
                    a_known = a.lower() in self.vocab or a.lower() in SHORT_WORDS
                    b_known = b.lower() in self.vocab or b.lower() in SHORT_WORDS
                    if b[0].islower() and (
                        (joined in self.vocab and not (a_known and b_known))
                        or (not a_known and not b_known and len(a) >= 3)
                    ):
                        out[-1] = a + b + tail
                        continue
            out.append(part)
        return re.sub(r"\s{2,}", " ", " ".join(out)).strip()


# ---------------------------------------------------------------------------
# Faculty directory (built from Faculty_Details + Contact PDFs, read-only)
# ---------------------------------------------------------------------------
class FacultyDirectory:
    TITLE_WORDS = {"prof", "professor", "dr", "mr", "ms", "mrs", "shri", "smt"}

    def __init__(self, pdf_dir, repair):
        self.repair = repair
        self.people = []
        self.office = None
        self.faculty_file = None
        self.contact_file = None
        try:
            self._load(pdf_dir)
        except Exception as exc:  # never break the app because of this layer
            print(f"[PRECISION] Faculty directory not built: {exc}")
        self.token_df = Counter()
        for p in self.people:
            for t in p["tokens"]:
                self.token_df[t] += 1

    # -- loading -----------------------------------------------------------
    def _find_pdf(self, pdf_dir, *keywords):
        for root, _, files in os.walk(pdf_dir):
            for f in sorted(files):
                low = f.lower()
                if low.endswith(".pdf") and all(k in low for k in keywords):
                    return os.path.join(root, f)
        return None

    def _load(self, pdf_dir):
        fac_pdf = self._find_pdf(pdf_dir, "faculty")
        con_pdf = self._find_pdf(pdf_dir, "contact")

        if fac_pdf:
            self.faculty_file = os.path.basename(fac_pdf)
            self._parse_faculty_pdf(fac_pdf)
        if con_pdf:
            self.contact_file = os.path.basename(con_pdf)
            self._parse_contact_pdf(con_pdf)

    def _parse_faculty_pdf(self, path):
        doc = fitz.open(path)
        lines = []
        for page_no, page in enumerate(doc, start=1):
            for ln in page.get_text("text").split("\n"):
                ln = re.sub(r"\s+", " ", ln).strip()
                if ln:
                    lines.append((page_no, ln))

        header = "department of computer science and engineering"
        lines = [(p, l) for p, l in lines if l.lower() != header]

        i = 0
        while i < len(lines):
            page, name = lines[i]
            # A record = Name, Designation, Qualification..., Email
            j = i + 1
            if j >= len(lines):
                break
            designation = lines[j][1]
            j += 1
            qual = []
            email = None
            while j < len(lines):
                txt = lines[j][1]
                if "(at)" in txt.lower() or "@" in txt:
                    email = txt
                    break
                qual.append(txt)
                j += 1
            if not email:
                break
            self._add_person(name, designation, " ".join(qual), email, page)
            i = j + 1

    def _add_person(self, name, designation, qualification, email_line, page):
        emails = [pretty_email(e) for e in email_line.split(",") if e.strip()]
        tokens = [w for w in WORD_RE.findall(name.lower()) if w not in self.TITLE_WORDS and len(w) >= 2]
        self.people.append({
            "name": name.strip(),
            "designation": designation.strip(),
            "qualification": qualification.strip(),
            "emails": emails,
            "room": None,
            "tokens": tokens,
            "page": page,
        })

    def _parse_contact_pdf(self, path):
        doc = fitz.open(path)
        for page in doc:
            try:
                tables = page.find_tables()
            except Exception:
                continue
            for tab in tables:
                for row in tab.extract():
                    if not row or len(row) < 4:
                        continue
                    sno, name, email, room = row[0], row[1], row[2], row[3]
                    if not sno or not str(sno).strip().isdigit() or not name:
                        continue
                    name_txt = self.repair.fix(re.sub(r"\s+", " ", name).strip())
                    room_txt = re.sub(r"\s*\|\s*", " | ", re.sub(r"\s+", " ", room or "")).strip(" |")
                    person = self._match_contact_name(name_txt)
                    if person is not None:
                        if room_txt:
                            person["room"] = room_txt
                    else:
                        tokens = [w for w in WORD_RE.findall(name_txt.lower()) if len(w) >= 2]
                        raw_email = re.sub(r"\s+", "", email or "")
                        self.people.append({
                            "name": name_txt, "designation": "", "qualification": "",
                            "emails": [pretty_email(raw_email)] if raw_email else [],
                            "room": room_txt or None, "tokens": tokens, "page": 1,
                        })

        # Department office block (text after the table)
        full = "\n".join(p.get_text("text") for p in doc)
        m = re.search(r"Faculty Assistants\s*\n(.*?)$", full, re.S | re.I)
        if m:
            block = [re.sub(r"\s+", " ", l).strip() for l in m.group(1).split("\n") if l.strip()]
            office = {"assistants": None, "room": None, "email": None}
            for l in block:
                low = l.lower()
                if low.startswith("room"):
                    office["room"] = l.split(":", 1)[-1].strip()
                elif low.startswith("email"):
                    office["email"] = pretty_email(l.split(":", 1)[-1])
                elif office["assistants"] is None:
                    office["assistants"] = l
            self.office = office

    def load_from_kb(self, kb):
        """Fallback / gap filler: read the contact table rows stored in the compiled KB."""
        added = 0
        for sec in kb.get("sections", []):
            if "contact" not in sec["source_file"].lower():
                continue
            for ln in sec["lines"]:
                row = parse_contact_row(ln)
                if not row:
                    continue
                name = self.repair.fix(row["name"])
                person = self._match_contact_name(name)
                if person is None:
                    toks = [w for w in WORD_RE.findall(name.lower()) if w not in self.TITLE_WORDS and len(w) >= 2]
                    person = {"name": name, "designation": "", "qualification": "", "emails": [],
                              "room": None, "tokens": toks, "page": sec["page"]}
                    self.people.append(person)
                    added += 1
                if not person["room"] and row["room"]:
                    person["room"] = row["room"]
                if not person["emails"] and row["email"]:
                    person["emails"] = [row["email"]]
                if not self.contact_file:
                    self.contact_file = sec["source_file"]
        if added:
            self.token_df = Counter(t for p in self.people for t in p["tokens"])
        return added

    def _match_contact_name(self, name_txt):
        toks = [w for w in WORD_RE.findall(name_txt.lower()) if len(w) >= 1]
        if not toks:
            return None
        first, last = toks[0], toks[-1]
        for p in self.people:
            ptoks = WORD_RE.findall(p["name"].lower())
            if ptoks and ptoks[-1] == last and ptoks[0][:3] == first[:3]:
                return p
        return None

    # -- querying ------------------------------------------------------------
    def match(self, query, known_vocab=frozenset()):
        """Return the list of best matching people (empty if nobody is named)."""
        if not self.people:
            return []
        q_words = WORD_RE.findall(query.lower())
        q_set = set(q_words)
        person_cue = bool(re.search(
            r"\b(prof|professor|dr|sir|ma'?am|madam|mr|mrs|ms|faculty|teacher)\b", query, re.I))

        scored = []
        for p in self.people:
            score = 0.0
            hits = 0
            for t in set(p["tokens"]):
                if t in q_set:
                    # short tokens (e.g. "dev", "b") are only trusted with a person cue
                    if len(t) <= 3 and not person_cue:
                        continue
                    hits += 1
                    score += 1.0 / self.token_df.get(t, 1)
            if hits:
                scored.append((hits, score, p))

        if not scored:
            # light fuzzy match for misspelt names (only words that are not normal KB words)
            all_tokens = {t: None for p in self.people for t in p["tokens"] if len(t) >= 4}
            for w in q_words:
                if len(w) < 5 or w in known_vocab or w in GENERIC_WORDS:
                    continue
                close = difflib.get_close_matches(w, list(all_tokens), n=1, cutoff=0.85)
                if close:
                    for p in self.people:
                        if close[0] in p["tokens"]:
                            scored.append((1, 1.0 / self.token_df.get(close[0], 1), p))
        if not scored:
            return []

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_hits, best_score = scored[0][0], scored[0][1]
        best = [p for h, s, p in scored if h == best_hits and abs(s - best_score) < 1e-9]
        return best

    def match_role(self, query):
        """'who is the dean', 'list of assistant professors', 'faculty list' ..."""
        q = query.lower()
        if not self.people:
            return None, []
        if re.search(r"\b(all|list|names?|every)\b.*\bfaculty\b|\bfaculty\s+(?:list|members?|names?)\b|\ball\s+professors\b", q):
            return "Faculty Members", list(self.people)

        roles = [
            ("associate dean", r"\bassociate\s+dean\b"),
            ("dean", r"\bdean\b"),
            ("vice president", r"\bvice[\s-]*president\b"),
            ("chairperson", r"\b(chair\s*person|chairman|chair|hod|head\s+of\s+(?:the\s+)?department)\b"),
            ("assistant professor", r"\bassistant\s+professors?\b"),
            ("associate professor", r"\bassociate\s+professors?\b"),
            ("professor", r"\bprofessors\b"),
        ]
        for role, pat in roles:
            if not re.search(pat, q):
                continue
            if role == "professor":
                found = [p for p in self.people if p["designation"].split("|")[0].strip().lower() == "professor"]
            elif role == "dean":
                found = [p for p in self.people
                         if re.search(r"(?<!associate )\bdean\b", p["designation"].lower())]
            elif role == "chairperson" and not re.search(r"\b(former|previous|ex)\b", q):
                found = [p for p in self.people if "chairperson" in p["designation"].lower()
                         and "former" not in p["designation"].lower()]
                if not found:
                    former = [p for p in self.people if "former chairperson" in p["designation"].lower()]
                    return "Current Chairperson", former
            else:
                found = [p for p in self.people if role in p["designation"].lower()]
            return role.title(), found
        return None, []


def clean_email(raw):
    """'pranabmu huri(at)cs . sau ac in . .' -> 'pranabmuhuri@cs.sau.ac.in' (best effort)."""
    if not raw:
        return ""
    e = re.sub(r"\(\s*\.?\s*a\s*t\s*\)|\(at\)", "@", raw, flags=re.I)
    if "@" not in e:
        return raw.strip()
    local, domain = e.split("@", 1)
    local = re.sub(r"\s+", "", local).strip(".")
    letters = re.sub(r"[^a-z]", "", domain.lower())
    known = {"cssauacin": "cs.sau.ac.in", "sauacin": "sau.ac.in", "sauint": "sau.int",
             "ieeeorg": "ieee.org", "gmailcom": "gmail.com"}
    dom = known.get(letters) or re.sub(r"\s+", "", domain).strip(".").replace("..", ".")
    return f"{local}@{dom}"


def parse_contact_row(line):
    """Parse one contact-table row in any of the compiled formats."""
    if "|" not in line:
        return None
    cells = [c.strip() for c in line.split("|")]
    name = email = room = None
    for c in cells:
        key, _, val = c.partition(":")
        k = re.sub(r"[^a-z]", "", key.lower())
        if k == "name":
            name = val.strip()
        elif k.startswith("email"):
            email = val.strip()
        elif k.startswith("room"):
            room = val.strip()
    if name is None:
        if len(cells) >= 3 and re.match(r"^[\d:\s]+$", cells[0]) and ("(at)" in cells[2] or "@" in cells[2]):
            name, email = cells[1], cells[2]
            room = " | ".join(cells[3:]) if len(cells) > 3 else None
        else:
            return None
    # rows merged by the PDF extractor keep the newer value after ': '
    pick = lambda v: v.split(": ")[-1].strip() if v and ": " in v else (v or "").strip()
    name, email, room = pick(name), pick(email), pick(room)
    if room:
        room = re.sub(r"\s*[—|]\s*", " | ", room).strip(" |")
    if not name or not re.search(r"[A-Za-z]{2}", name) or re.search(r"\bname\b", name, re.I):
        return None
    return {"name": re.sub(r"\s+", " ", name), "email": clean_email(email), "room": room or None}


# ---------------------------------------------------------------------------
# Fee table (built from Fee_Structure PDF, read-only)
# ---------------------------------------------------------------------------
class FeeTable:
    # How each department programme is written inside the fee table
    PROGRAM_ROW_PATTERNS = {
        "mca": r"\bmca\b",
        "msc": r"\bm\.?\s*sc\.?\s+computer\s+science",
        "mtech": r"\bm\.?\s*tech\.?\s+computer\s+science",
        "btech_cse": r"\bb\.?\s*tech\.?\s+computer\s+science",
        "dual_degree": r"\bdual\s+degree\b",
        "btech_mnc": r"\bb\.?\s*tech\.?\s+mathematics",
        "phd": r"\bph\.?\s*d\.?\s+in\b[^|]*computer\s+science",
    }

    def __init__(self, pdf_dir, repair, kb=None):
        self.repair = repair
        self.rows = []          # {"label", "values": [(column, amount)], "group"}
        self.notes = []
        self.file = None
        try:
            self._load(pdf_dir)
        except Exception as exc:
            print(f"[PRECISION] Fee table not built: {exc}")
        # "Other Details" notes (exchange rate, installments, refunds) from the compiled KB
        for sec in (kb or {}).get("sections", []):
            if self.file and sec["source_file"] == self.file:
                grab = False
                for ln in sec["lines"]:
                    if ln.strip().lower().startswith("other details"):
                        grab = True
                        continue
                    if grab and len(ln.split()) >= 5:
                        self.notes.append(ln.strip())

    def _clean(self, cell):
        if cell is None:
            return ""
        return self.repair.fix(re.sub(r"\s+", " ", str(cell)).strip())

    def _load(self, pdf_dir):
        path = None
        for root, _, files in os.walk(pdf_dir):
            for f in files:
                if f.lower().endswith(".pdf") and "fee" in f.lower():
                    path = os.path.join(root, f)
        if not path:
            return
        self.file = os.path.basename(path)
        doc = fitz.open(path)
        header = ["Fee Component", "Amount (USD)"]
        group = "General Fees"
        for page in doc:
            for tab in page.find_tables():
                for raw in tab.extract():
                    cells = [self._clean(c) for c in raw]
                    if not any(cells):
                        continue
                    label, values = cells[0], cells[1:]
                    nonempty_vals = [v for v in values if v]
                    if label.lower().startswith(("fee component", "programme", "program me", "phd programm")) and \
                            any("amount" in v.lower() or "mode" in v.lower() for v in nonempty_vals):
                        header = cells
                        if label.lower().startswith("phd"):
                            group = "PhD Programmes"
                        continue
                    if raw[1:] and all(c is None for c in raw[1:]) or (label and not nonempty_vals and label.endswith("Programmes")):
                        group = label
                        continue
                    if not nonempty_vals and self.rows and label:
                        # continuation of the previous row (label wrapped onto next page)
                        self.rows[-1]["label"] += " " + label
                        continue
                    pairs = []
                    for idx, v in enumerate(values):
                        if v:
                            col = header[idx + 1] if idx + 1 < len(header) else "Amount (USD)"
                            pairs.append((col, v))
                    self.rows.append({"label": label, "values": pairs, "group": group})

    def exchange_rate(self):
        for n in self.notes:
            m = re.search(r"USD\s*1\s*=\s*INR\s*([\d.]+)", n)
            if m:
                return float(m.group(1)), n
        return None, None

    def rows_for_program(self, prog):
        pat = self.PROGRAM_ROW_PATTERNS.get(prog)
        if not pat:
            return []
        out = []
        for r in self.rows:
            label = r["label"]
            if prog != "dual_degree":
                # "Dual Degree BTech – MTech ..." must not count as an M.Tech / B.Tech row
                label = re.sub(r"dual\s+degree\s+b\.?\s*tech\s*[–-]\s*m\.?\s*tech", " ", label, flags=re.I)
            if re.search(pat, label, re.I):
                out.append(r)
        return out

    def rows_for_tokens(self, q_tokens):
        best, best_score = [], 0
        for r in self.rows:
            r_toks = set(content_tokens(r["label"]))
            s = len(r_toks & q_tokens)
            if s > best_score:
                best, best_score = [r], s
            elif s == best_score and s > 0:
                best.append(r)
        return best if best_score > 0 else []


# ---------------------------------------------------------------------------
# Section focusing (sub-section / line / semester level)
# ---------------------------------------------------------------------------
SEMESTER_LINE_RE = re.compile(
    r"^\s*(?:semester\s*:?\s*)?(\d)\s*(?:st|nd|rd|th)\b|^\s*semester\s*[-:]?\s*(\d|[ivx]+)\s*$",
    re.IGNORECASE,
)


def requested_semester(query):
    q = query.lower()
    m = re.search(r"\b(?:semester|sem)\s*[-:]?\s*(\d|[ivx]{1,4})\b", q)
    if m:
        v = m.group(1)
        return int(v) if v.isdigit() else ORDINAL_WORDS.get(v)
    m = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|1st|2nd|3rd|4th|5th|6th|7th|8th)\s+(?:semester|sem)\b", q)
    if m:
        return ORDINAL_WORDS[m.group(1)]
    return None


def split_blocks(sec):
    """Split a section into (heading, [lines]) blocks using the ##SUBSECTION## markers."""
    blocks = [[sec["heading"], []]]
    for ln in sec["lines"]:
        if ln.startswith("##SUBSECTION##"):
            blocks.append([ln.replace("##SUBSECTION##", "").strip(), []])
        else:
            blocks[-1][1].append(ln.strip())
    return blocks


def clean_table_row(line, header):
    """'Course Title: Data Structures | Credits: 4' -> ['Data Structures', '4']"""
    cells = [c.strip() for c in line.split("|")]
    out = []
    for c in cells:
        if not c:
            continue
        stripped = None
        for h in header:
            if h and c.startswith(h + ":"):
                stripped = c[len(h) + 1:].strip()
                break
        if stripped is None and ": " in c:
            # two table rows merged by the PDF extractor -> keep the newer value
            stripped = c.split(": ")[-1].strip()
        out.append(stripped if stripped is not None else c)
    return [o for o in out if o and o not in {"-", "--", "- -"}]


def is_fragment(line):
    """Broken table-cell pieces such as 'Programm', '3-0-2', 'ing and'."""
    if "|" in line:
        return False
    if SEMESTER_LINE_RE.match(line):
        return False
    return len(line.split()) < 6 and not line.rstrip().endswith((".", ":"))


# ---------------------------------------------------------------------------
# The precision layer used by PrologInferenceEngine
# ---------------------------------------------------------------------------
PROGRAM_LABELS = {
    "mca": "MCA", "msc": "M.Sc. (Computer Science)", "mtech": "M.Tech. (CSE)",
    "btech_cse": "B.Tech. (CSE)", "btech_mnc": "B.Tech. (Mathematics & Computing)",
    "dual_degree": "Dual Degree B.Tech.–M.Tech. (CSE)", "phd": "PhD (CSE)", "bsc": "B.Sc.",
}

EMAIL_RE = re.compile(r"\b(e-?mail|mail|email\s*id|mail\s*id|e-mail\s*address)\b", re.I)
ROOM_RE = re.compile(r"\b(room|office|cabin|chamber|sits?|seated|located|location|where\s+(?:does|do|can|is|to)|meet)\b", re.I)
DESIG_RE = re.compile(r"\b(designation|position|post|role|rank|title)\b|^\s*who\s+is\b", re.I)
QUAL_RE = re.compile(r"\b(qualifications?|ph\.?\s*d|degree|education(?:al)?|doctorate|studied|alma\s+mater)\b", re.I)
PHONE_RE = re.compile(r"\b(phone|mobile|telephone|cell\s*(?:no|number)?|contact\s+(?:no|number)|call)\b", re.I)
CONTACT_RE = re.compile(r"\b(contact|reach|get\s+in\s+touch)\b", re.I)


def good_heading(h):
    h = (h or "").strip()
    return len(h) >= 8 and len(h.split()) >= 2 and (h[0].isupper() or h[0].isdigit()) and \
        not re.match(r"^(ent|tive|ing|ion|tion)\b", h, re.I)


def section_label(sec, progs=None):
    p = set(sec["programs"]) - {"general"}
    if progs:
        p &= set(progs) | ({"btech_cse"} if "btech_mnc" in progs else set())
    if "btech_cse" in p and "math" in sec["source_file"].lower():
        p = (p - {"btech_cse"}) | {"btech_mnc"}
    return program_label(p)


NOT_FOUND = "__NOT_FOUND__"

# Common short forms / informal words -> the wording used in the knowledge base.
# (pattern, replacement) - applied as whole words, case-insensitive, before anything else.
TERM_MAP = [
    # programmes
    (r"m\s*&\s*c|mnc", "mathematics and computing"),
    (r"maths?|mathematic", "mathematics"),
    (r"comp\.?\s*sci\.?|compsci|comp\s+science", "computer science"),
    (r"b\s*\.?\s*e\.?(?=\s+(?:cse|computer|degree))", "btech"),
    (r"masters?\s+(?:in|of)\s+computer\s+applications?", "mca"),
    (r"bachelors?\s+(?:in|of)\s+technology", "btech"),
    (r"masters?\s+(?:in|of)\s+technology", "mtech"),
    (r"doctorate|doctoral|ph\.?\s*d\.?", "phd"),
    (r"pg", "postgraduate"),
    (r"ug", "undergraduate"),
    # subjects
    (r"ai\s*(?:&|and|/)\s*ml|aiml", "artificial intelligence and machine learning"),
    (r"ai", "artificial intelligence"),
    (r"ml", "machine learning"),
    (r"dl", "deep learning"),
    (r"nlp", "natural language processing"),
    (r"dbms|rdbms|db", "database systems"),
    (r"os", "operating systems"),
    (r"cn", "computer networks"),
    (r"daa", "design and analysis of algorithms"),
    (r"dsa|ds", "data structures"),
    (r"oops?|oo", "object oriented"),
    (r"coa", "computer organization"),
    (r"se", "software engineering"),
    (r"toc", "automata theory"),
    (r"devops", "dev ops"),
    (r"cyber\s*sec|cybersec", "cyber security"),
    # admin / general words
    (r"dept\.?|deptt\.?", "department"),
    (r"sem", "semester"),
    (r"sems", "semesters"),
    (r"yrs?", "years"),
    (r"elig|eligiblity|eligibilty", "eligibility"),
    (r"reqd|req", "required"),
    (r"cut\s*-?\s*off|cutoff", "minimum marks"),
    (r"intake|capacity", "seats"),
    (r"syllabi", "syllabus"),
    (r"subj|subjs", "subjects"),
    (r"tution|tuition\s+fee", "fees"),
    (r"ph\.?\s*no\.?|mob(?:ile)?\s*no\.?|contact\s*no\.?", "phone number"),
    (r"e-?mail\s*(?:id|address)?|mail\s*id", "email"),
    (r"addr", "address"),
    (r"hod|h\.o\.d\.?", "head of department"),
    (r"profs?", "professor"),
    (r"asst\.?\s*professor|asst\.?\s*prof", "assistant professor"),
    (r"assoc\.?\s*professor|assoc\.?\s*prof", "associate professor"),
    (r"info", "information"),
    (r"pls|plz", "please"),
    (r"abt", "about"),
    (r"wat|wht", "what"),
    (r"u", "you"),
]
TERM_MAP = [(re.compile(r"(?<![\w.&])(?:" + pat + r")(?![\w&])", re.I), rep) for pat, rep in TERM_MAP]


SHORT_FORMS = {"maths", "math", "devops", "compsci", "dbms", "rdbms", "aiml", "deptt", "syllabi", "subjs",
               "reqd", "cutoff", "tution", "addr", "asst", "assoc", "profs"}


def normalize_terms(query):
    """'maths and computing' -> 'mathematics and computing', 'dbms credits' -> 'database systems credits'."""
    out = query
    for rx, rep in TERM_MAP:
        out = rx.sub(rep, out)
    return re.sub(r"\s{2,}", " ", out).strip()


# Words the department questions are built from (used to fix spelling mistakes)
DOMAIN_WORDS = {
    "eligibility", "eligible", "criteria", "qualification", "qualifications", "admission", "admissions",
    "syllabus", "curriculum", "structure", "semester", "semesters", "subjects", "courses", "credits",
    "elective", "electives", "entrance", "examination", "duration", "seats", "matrix", "scholarship",
    "hostel", "accommodation", "contact", "details", "email", "phone", "number", "professor",
    "professors", "faculty", "designation", "department", "fees", "fee", "tuition", "installment",
    "refund", "exchange", "registration", "deposit", "loan", "interest", "chairperson", "associate",
    "assistant", "dean", "programme", "program", "minimum", "maximum", "percentage", "marks",
    "negative", "questions", "specialization", "specialisation", "tracks", "laboratory", "labs",
    "research", "established", "vision", "mission", "overview", "process", "mode", "modes",
}


def program_label(progs):
    return ", ".join(PROGRAM_LABELS.get(p, p.upper()) for p in sorted(progs)) if progs else ""


class PrecisionLayer:
    def __init__(self, kb, compiler, pdf_dir):
        self.kb = kb or {"sections": [], "faculty": []}
        self.compiler = compiler
        self.repair = TextRepair(self.kb)
        self.pdf_dir = pdf_dir
        self.directory = FacultyDirectory(pdf_dir, self.repair)
        kb_people = self.directory.load_from_kb(self.kb)
        self.fees = FeeTable(pdf_dir, self.repair, self.kb)

        # token weights for line-level matching
        df = Counter()
        n = 0
        for sec in self.kb.get("sections", []):
            for ln in sec["lines"]:
                n += 1
                for t in set(content_tokens(ln)):
                    df[t] += 1
        self.n_lines = max(1, n)
        self.df = df
        self.vocab = set(self.repair.vocab)
        self.spell_vocab = (self.vocab | DOMAIN_WORDS | GENERIC_WORDS | PROGRAM_WORDS |
                            {t for p in self.directory.people for t in p["tokens"]})
        self._spell_list = sorted(w for w in self.spell_vocab if len(w) >= 4)
        print("=" * 70)
        print(f"[PRECISION v{PRECISION_VERSION}] Ready: {len(self.directory.people)} faculty profiles, "
              f"{len(self.fees.rows)} fee rows.")
        print(f"[PRECISION v{PRECISION_VERSION}] PDF folder : {os.path.abspath(pdf_dir)} "
              f"({'found' if os.path.isdir(pdf_dir) else 'NOT FOUND'})")
        print(f"[PRECISION v{PRECISION_VERSION}] Faculty PDF: {self.directory.faculty_file or 'NOT FOUND'} | "
              f"Contact PDF: {self.directory.contact_file or 'NOT FOUND'} | from KB rows: {kb_people}")
        if not self.directory.people:
            print(f"[PRECISION v{PRECISION_VERSION}] WARNING: no faculty profiles could be built!")
        print("=" * 70)

    def correct_spelling(self, query):
        """'what is the eligibilty of mca' -> 'what is the eligibility of mca' (KB words only)."""
        def fix(m):
            word = m.group(0)
            low = word.lower()
            if len(low) < 4 or low in self.spell_vocab or low in SHORT_FORMS:
                return word
            close = difflib.get_close_matches(low, sorted(DOMAIN_WORDS), n=1, cutoff=0.75) or \
                difflib.get_close_matches(low, self._spell_list, n=1, cutoff=0.82)
            return close[0] if close else word
        fixed = normalize_terms(re.sub(r"[A-Za-z]+", fix, normalize_terms(query)))
        if fixed != query:
            print(f"[PRECISION] Query normalised: '{query}' -> '{fixed}'")
        return fixed

    def final_filter(self, query, response):
        """Last line of defence: a table answer for a question that names a person -> only that row."""
        if not response or not response.get("found"):
            return response
        ans = response.get("answer", "")
        lines = ans.split("\n")
        rows = [(i, parse_contact_row(re.sub(r"\s+—\s+", " | ", l.lstrip("•▸ ").strip()))) for i, l in enumerate(lines)]
        rows = [(i, r) for i, r in rows if r]
        if len(rows) < 2:
            return response
        q_words = {w for w in WORD_RE.findall(query.lower()) if len(w) >= 3 and w not in GENERIC_WORDS}
        hits = []
        for i, r in rows:
            name_toks = {w for w in WORD_RE.findall(self.repair.fix(r["name"]).lower()) if len(w) >= 3}
            squashed = re.sub(r"[^a-z]", "", r["name"].lower())
            if q_words & name_toks or any(len(w) >= 5 and w in squashed for w in q_words):
                hits.append(r)
        if not hits or len(hits) == len(rows):
            return response
        fields = self._wanted_person_fields(query)
        out = []
        for r in hits:
            p = self.directory._match_contact_name(self.repair.fix(r["name"])) or {
                "name": self.repair.fix(r["name"]), "designation": "", "qualification": "",
                "emails": [r["email"]] if r["email"] else [], "room": r["room"]}
            out.append(f"**{p['name']}**")
            out.extend(self._person_lines(p, fields))
            out.append("")
        response = dict(response)
        response["answer"] = "\n".join(out).strip()
        trace = dict(response.get("rag_trace") or {})
        trace["status"] = (trace.get("status", "") + f" | precision v{PRECISION_VERSION}: filtered to "
                           f"{', '.join(h['name'] for h in hits)}").strip(" |")
        response["rag_trace"] = trace
        return response

    def w(self, tok):
        return math.log(1.0 + self.n_lines / (1.0 + self.df.get(tok, 0)))

    # ------------------------------------------------------------------
    # Response helper (same JSON shape the UI already understands)
    # ------------------------------------------------------------------
    def _response(self, answer, file, section, page=1, goal_str="", status="", conf=99.0, snippet=None):
        return {
            "found": True,
            "answer": answer.strip(),
            "confidence": conf,
            "sources": [{"file": file or "Knowledge Base", "category": "General", "page": page, "section": section}],
            "rag_trace": {
                "expanded_query": goal_str,
                "status": status,
                "retrieved_chunks": [{
                    "chunk_id": 0, "file": file or "Knowledge Base", "category": "General",
                    "section": section, "page": page, "score": 1.0,
                    "snippet": (snippet or answer)[:260],
                }],
            },
        }

    # ------------------------------------------------------------------
    # 1. Direct answers: faculty member, role, department office, fees
    # ------------------------------------------------------------------
    def try_direct(self, query, goal):
        topics = goal.get("topics", set())
        people = self.directory.match(query, self.vocab)
        if people and not ({"fees", "course_structure"} & topics and not CONTACT_RE.search(query)):
            return self._answer_people(query, people)

        if not goal.get("programs"):
            role, found = self.directory.match_role(query)
            if role:
                return self._answer_role(query, role, found)
            office = self._answer_office(query)
            if office:
                return office

        m = re.search(r"\b(hostel|mess|accommodation|loan)\b", query, re.I)
        if m:
            kw = "loan" if m.group(1).lower() == "loan" else "hostel"
            res = self._file_scoped_answer(query, kw)
            if res:
                return res

        if "fees" in topics or re.search(r"\b(fees?|tuition|cost|charges?|installments?|refunds?|exchange\s+rate)\b", query, re.I):
            return self._answer_fees(query, goal)
        return None

    def _file_scoped_answer(self, query, kw):
        secs = [s for s in self.kb["sections"] if kw in s["source_file"].lower()]
        if not secs:
            return None
        secs.sort(key=lambda s: s["sec_id"])
        title = "Hostel" if kw == "hostel" else "Student Loan"

        if CONTACT_RE.search(query) or PHONE_RE.search(query) or EMAIL_RE.search(query):
            anchor = None
            for i, s in enumerate(secs):
                if re.search(r"queries,?\s+please\s+contact|contact\s+person", s["heading"], re.I):
                    anchor = i
                    break
            if anchor is not None:
                out = [f"**{title} — Contact**"]
                seen = set()
                for s in secs[anchor:anchor + 3]:
                    texts = ([] if s is secs[anchor] else [s["heading"]]) + s["lines"]
                    for t in texts:
                        t = t.replace("##SUBSECTION##", "").strip()
                        key = re.sub(r"\W+", "", t.lower())
                        if t and key not in seen:
                            seen.add(key)
                            out.append(self._format_line(t))
                    if s is not secs[anchor] and not re.search(r"(ph\.?\s*no|email|mob)", s["heading"] + " ".join(s["lines"]), re.I):
                        break
                return self._response("\n".join(out), secs[anchor]["source_file"], secs[anchor]["heading"],
                                      secs[anchor]["page"], goal_str=f"?- contact({kw}, Details).",
                                      status="Prolog Unification Succeeded: contact", conf=97.5)

        ranked = [(1.0, s) for s in secs]
        fact = self._fact_lines(query, {"programs": set()}, ranked, force=True, min_coverage=0.5, flat=True)
        if not fact:
            return None
        heading, lines, used = fact
        if len(heading.split()) > 8:
            heading = title
        return self._response(f"**{self.repair.fix(heading)}**\n" + "\n".join(lines), used[0]["source_file"],
                              used[0]["heading"], used[0]["page"], goal_str=f"?- fact({kw}, Line).",
                              status="Prolog Unification Succeeded: exact line", conf=96.0)

    def _wanted_person_fields(self, query):
        fields = []
        if PHONE_RE.search(query):
            fields.append("phone")
        if EMAIL_RE.search(query):
            fields.append("email")
        if ROOM_RE.search(query):
            fields.append("room")
        if QUAL_RE.search(query):
            fields.append("qualification")
        if DESIG_RE.search(query):
            fields.append("designation")
        if not fields and CONTACT_RE.search(query):
            fields = ["email", "room"]
        if "phone" in fields:
            for f in ("email", "room"):
                if f not in fields:
                    fields.append(f)
        return fields or ["designation", "qualification", "email", "room"]

    def _person_lines(self, p, fields):
        lines = []
        for f in fields:
            if f == "phone":
                lines.append("• **Phone:** Not listed in the department knowledge base")
            elif f == "email":
                lines.append(f"• **Email:** {', '.join(p['emails']) if p['emails'] else 'Not listed'}")
            elif f == "room":
                lines.append(f"• **Room No.:** {p['room'] or 'Not listed'}")
            elif f == "designation":
                lines.append(f"• **Designation:** {p['designation'] or 'Not listed'}")
            elif f == "qualification":
                lines.append(f"• **Qualification:** {p['qualification'] or 'Not listed'}")
        return lines

    def _answer_people(self, query, people):
        fields = self._wanted_person_fields(query)
        out = []
        if len(people) > 1:
            out.append(f"{len(people)} faculty members match your query:")
            out.append("")
        for p in people:
            if len(people) == 1 and len(fields) == 1 and fields[0] != "phone":
                f = fields[0]
                value = {
                    "email": ", ".join(p["emails"]) or "not listed",
                    "room": p["room"] or "not listed",
                    "designation": p["designation"] or "not listed",
                    "qualification": p["qualification"] or "not listed",
                }[f]
                label = {"email": "Email", "room": "Room No.", "designation": "Designation",
                         "qualification": "Qualification"}[f]
                out.append(f"**{p['name']}**")
                out.append(f"• **{label}:** {value}")
            else:
                out.append(f"**{p['name']}**")
                out.extend(self._person_lines(p, fields))
            out.append("")
        names = ", ".join(p["name"] for p in people)
        file = self.directory.faculty_file if any(f in fields for f in ("designation", "qualification", "email")) \
            else (self.directory.contact_file or self.directory.faculty_file)
        atom = "_".join(WORD_RE.findall(people[0]["name"].lower()))
        return self._response(
            "\n".join(out), file, names, people[0].get("page", 1),
            goal_str=f"?- faculty({atom}, [{', '.join(fields)}]).",
            status=f"Prolog Unification Succeeded: Bound to {names} -> fields {fields}",
            conf=99.4,
        )

    def _answer_role(self, query, role, found):
        if role == "Current Chairperson":
            txt = "The current Chairperson / Head of Department is not listed in the department knowledge base."
            if found:
                txt += "\n\n" + "\n".join(f"• **Former Chairperson:** {p['name']} ({p['designation']})" for p in found)
            return self._response(txt, self.directory.faculty_file, "Chairperson",
                                  goal_str="?- designation(Person, 'chairperson').",
                                  status="Prolog Unification: no current chairperson fact; former chairperson bound", conf=90.0)
        if not found:
            return self._response(
                f"No faculty member with the designation **{role}** is listed in the department knowledge base.",
                self.directory.faculty_file, role, goal_str=f"?- designation(Person, '{role}').",
                status="Prolog Unification Failed for role", conf=80.0)
        fields = []
        if EMAIL_RE.search(query) or CONTACT_RE.search(query):
            fields.append("email")
        if ROOM_RE.search(query) or CONTACT_RE.search(query):
            fields.append("room")
        out = []
        if re.search(r"\bhow\s+many\b|\bnumber\s+of\b|\bcount\b", query, re.I):
            out.append(f"There are **{len(found)}** {role.lower()}{'s' if len(found) != 1 and not role.endswith('s') else ''} "
                       f"listed in the department.")
            out.append("")
        if len(found) == 1 and not fields:
            p = found[0]
            out.append(f"**{p['name']}** — {p['designation']}")
        else:
            out.append(f"**{role}**")
            for p in found:
                extra = []
                if "email" in fields:
                    extra.append(", ".join(p["emails"]))
                if "room" in fields and p["room"]:
                    extra.append(f"Room {p['room']}")
                desig = "" if role != "Faculty Members" and role.lower() in p["designation"].lower() and \
                    p["designation"].lower() == role.lower() else f" — {p['designation']}"
                out.append(f"• **{p['name']}**{desig}" + (f" — {' — '.join(extra)}" if extra else ""))
        return self._response("\n".join(out), self.directory.faculty_file, role,
                              goal_str=f"?- designation(Person, '{role.lower()}').",
                              status=f"Prolog Unification Succeeded: {len(found)} binding(s)", conf=98.5)

    def _answer_office(self, query):
        office = self.directory.office
        if not office or not CONTACT_RE.search(query) and not EMAIL_RE.search(query) and not PHONE_RE.search(query):
            return None
        if re.search(r"\b(hostel|loan|bank|mess|warden|student\s+services)\b", query, re.I):
            return None
        if not re.search(r"\b(department|dept|cse|cs|office|administration|admin|computer\s+science)\b", query, re.I):
            return None
        out = ["**Department of Computer Science & Engineering — Office Contact**"]
        if PHONE_RE.search(query):
            out.append("• **Phone:** Not listed in the department knowledge base")
        if office.get("email"):
            out.append(f"• **Email:** {office['email']}")
        if office.get("room"):
            out.append(f"• **Room No.:** {office['room']}")
        if office.get("assistants"):
            out.append(f"• **Faculty Assistants:** {office['assistants']}")
        return self._response("\n".join(out), self.directory.contact_file, "Faculty Assistants",
                              goal_str="?- department_office(Email, Room, Assistants).",
                              status="Prolog Unification Succeeded: department office", conf=98.0)

    def _answer_fees(self, query, goal):
        if not self.fees.rows or re.search(r"\b(hostel|mess|loan)\b", query, re.I):
            return None
        q = query.lower()
        q_toks = set(content_tokens(query)) - {"fee", "semester", "per", "structure", "amount"}
        rate, rate_note = self.fees.exchange_rate()
        want_inr = bool(re.search(r"\b(inr|rupees?|rs\.?|₹|indian\s+currency)\b", q))
        out = []

        # notes: installment / refund / non-SAARC / exchange rate
        note_hits = []
        for n in self.fees.notes:
            n_toks = set(content_tokens(n))
            if re.search(r"\b(installments?|refunds?|non[\s-]*saarc|exchange|conversion|dollar\s+rate)\b", q) and \
                    (q_toks & n_toks or ("exchange" in q and "exchange" in n.lower())):
                note_hits.append(n)
        if note_hits and not goal.get("programs"):
            out = ["**Fee — Other Details**"] + [f"• {n}" for n in note_hits]
            return self._response("\n".join(out), self.fees.file, "Other Details",
                                  goal_str="?- fee_note(Note).", status="Prolog Fee Unification Succeeded")

        def money(v):
            m = re.match(r"^(\d[\d,]*)(.*)$", v.strip())
            if not m:
                return v
            amt = int(m.group(1).replace(",", ""))
            txt = f"USD {amt:,}{m.group(2)}"
            if want_inr and rate:
                txt += f" (≈ INR {int(amt * rate):,})"
            return txt

        progs = set(goal.get("programs") or [])
        if progs:
            for prog in sorted(progs):
                rows = self.fees.rows_for_program(prog)
                if not rows:
                    continue
                r = rows[0]
                label = PROGRAM_LABELS.get(prog, prog.upper())
                if len(r["values"]) == 1:
                    out.append(f"The semester fee for **{label}** is **{money(r['values'][0][1])} per semester**.")
                else:
                    out.append(f"**{label} — Semester Fee**")
                    for col, v in r["values"]:
                        out.append(f"• **{col}:** {money(v)}")
            if not out:
                return None
        else:
            rows = self.fees.rows_for_tokens(q_toks) if q_toks else []
            if rows:
                for r in rows:
                    vals = "; ".join(f"{money(v)}" if col.lower().startswith("amount") else f"{col}: {money(v)}"
                                     for col, v in r["values"])
                    out.append(f"• **{r['label']}:** {vals}")
            else:
                # the user asked for the fee structure in general -> department programmes + general fees
                out.append("**Semester Fees — Department Programmes**")
                for prog in ["btech_cse", "btech_mnc", "dual_degree", "mca", "msc", "mtech", "phd"]:
                    rows = self.fees.rows_for_program(prog)
                    if rows:
                        vals = "; ".join(money(v) if col.lower().startswith("amount") else f"{col}: {money(v)}"
                                         for col, v in rows[0]["values"])
                        out.append(f"• **{PROGRAM_LABELS[prog]}:** {vals}")
                out.append("")
                out.append("**One-time / Other Fees**")
                for r in self.fees.rows:
                    if r["group"] == "General Fees":
                        out.append(f"• **{r['label']}:** {money(r['values'][0][1])}")
        if want_inr and rate_note:
            out.append("")
            out.append(f"_{rate_note.split(' The Conversion')[0]}_")
        return self._response("\n".join(out), self.fees.file, "Semester Fees by Programme",
                              goal_str=f"?- fee({list(progs) or '_'}, Amount).",
                              status="Prolog Fee Unification Succeeded", conf=99.2)

    # ------------------------------------------------------------------
    # 2. Narrow a retrieved section down to the exact answer
    # ------------------------------------------------------------------
    def focus(self, query, goal, ranked_sections):
        """ranked_sections: list of (score, sec) best first. Returns (answer_text, used_sections) or None."""
        if not ranked_sections:
            return None
        best = ranked_sections[0][1]
        progs = goal.get("programs") or set()
        topics = goal.get("topics") or set()
        label = section_label(best, progs) if progs else ""
        same_file = [s for _, s in ranked_sections if s["source_file"] == best["source_file"]]
        all_file_secs = [s for s in self.kb["sections"] if s["source_file"] == best["source_file"]
                         and (not progs or progs & s["programs"] or s["programs"] == {"general"})]

        # relevance gate for questions with no programme and no topic
        q_all = set(content_tokens(query))
        if not progs and not topics and q_all:
            sec_t = set(content_tokens(best["heading"] + " " + " ".join(best["lines"])))
            w_all = sum(self.w(t) for t in q_all)
            w_hit = sum(self.w(t) for t in q_all & sec_t)
            if w_hit / w_all < 0.34:
                return NOT_FOUND, [best]

        # (a) a specific semester
        sem = requested_semester(query)
        if sem:
            cs_secs = [s for s in all_file_secs if s["topic"] == "course_structure" and
                       any("|" in ln for ln in s["lines"]) and not re.search(r"entrance|syllabus", s["heading"], re.I)]
            if cs_secs:
                rows, notes, found_sems = self._course_rows(cs_secs)
                sem_rows = [r for s_no, r in rows if s_no == sem]
                title = f"{label + ' — ' if label else ''}Semester {sem} Courses"
                if sem_rows:
                    return f"**{title}**\n" + "\n".join(sem_rows), cs_secs
                if found_sems:
                    avail = ", ".join(str(x) for x in sorted(found_sems))
                    return (f"Semester {sem} is not listed for {label or 'this programme'} in the knowledge base. "
                            f"Available semesters: {avail}."), cs_secs

        # (b) a specific fact (total seats, duration of test, minimum marks, ...)
        fact = self._fact_lines(query, goal, ranked_sections)
        if fact:
            heading, lines, secs = fact
            heading = self.repair.fix(heading)
            title = heading if label and label.lower() in heading.lower() else f"{label + ' — ' if label else ''}{heading}"
            return f"**{title}**\n" + "\n".join(lines), secs

        # (c) the best section never mentions the asked words -> find the exact line anywhere in the KB
        q_toks = set(content_tokens(query))
        sec_toks = set(content_tokens(best["heading"])) | set(t for ln in best["lines"] for t in content_tokens(ln))
        if q_toks and not (q_toks & sec_toks):
            pool = []
            for sec in self.kb["sections"]:
                if progs and not (progs & sec["programs"]) and sec["programs"] != {"general"}:
                    continue
                blob = set(content_tokens(sec["heading"] + " " + " ".join(sec["lines"])))
                ov = len(q_toks & blob)
                if ov:
                    pool.append((ov, sec))
            pool.sort(key=lambda x: x[0], reverse=True)
            fact = self._fact_lines(query, goal, pool[:12], force=True, min_coverage=0.99, flat=True)
            if fact:
                heading, lines, secs = fact
                if len(heading.split()) > 8:
                    heading = secs[0]["doc_title"]
                lab = section_label(secs[0], progs or None)
                heading = self.repair.fix(heading)
                title = heading if lab and lab.split(" ")[0].lower() in heading.lower() else f"{lab + ' — ' if lab else ''}{heading}"
                return f"**{title}**\n" + "\n".join(lines), secs

        # (d) full course structure (cleaned table rows only)
        if "course_structure" in topics and best["topic"] == "course_structure" and \
                any("|" in ln for ln in best["lines"]):
            cs_secs = [s for s in same_file if s["topic"] == "course_structure" and any("|" in ln for ln in s["lines"])
                       and (not re.search(r"entrance|syllabus", s["heading"], re.I)
                            or re.search(r"entrance|syllabus", query, re.I))]
            # "electives" etc. -> only the section whose heading names it
            named = [s for s in cs_secs if q_toks & set(content_tokens(s["heading"])) - {"structure"}]
            if named:
                cs_secs = named
            rows, notes, found_sems = self._course_rows(cs_secs)
            if rows:
                head = self.repair.fix(cs_secs[0]["heading"]) if named else "Course Structure"
                out = [f"**{label + ' — ' if label else ''}{head}**"]
                cur = object()
                for s_no, r in rows:
                    if s_no != cur:
                        out.append("")
                        out.append(f"**Semester {s_no}**" if s_no else "**Other Courses**")
                        cur = s_no
                    out.append(r)
                for n in notes:
                    out.append(f"• _{n}_")
                return "\n".join(out), cs_secs

        # (e) best sub-section only
        return self._best_block(query, goal, best, label)

    # -- (a)/(c) course table rows ---------------------------------------
    def _format_course_row(self, cells):
        cells = [self.repair.fix(re.sub(r"(\s*-\s*){1,3}$", "", c).strip()) for c in cells]
        cells = [c for c in cells if c and not re.fullmatch(r"(?:semester\s*:?\s*)?\d\s*(?:st|nd|rd|th)", c, re.I)
                 and not re.fullmatch(r"[*\s]+", c)]
        code, name, ltp, credits, kind = None, [], None, None, None
        for c in cells:
            if re.fullmatch(r"[A-Z]{2,4}\s?-?\s?[\dA-Z*]{3}\s*-?", c) or re.fullmatch(r"[A-Z]{3}\s?\d\w\*?\s*-?", c):
                code = c.rstrip(" -")
            elif re.fullmatch(r"\d\s*[- ]\s*\d\s*[- ]\s*\d", c):
                ltp = "-".join(re.findall(r"\d", c))
            elif re.fullmatch(r"\d+(?:\.\d)?\*?", c):
                credits = c
            elif c.lower() in {"core", "elective"}:
                kind = c.title()
            else:
                name.append(c)
        if not name and not code:
            return None
        txt = " ".join(x for x in [code, " / ".join(name)] if x)
        bits = [txt]
        if ltp:
            bits.append(f"L-T-P {ltp}")
        if credits:
            bits.append(f"{credits} credits")
        line = "• " + " — ".join(bits)
        if kind:
            line += f" ({kind})"
        return line

    def _course_rows(self, secs):
        rows, notes, seen, found = [], [], set(), set()
        for sec in sorted(secs, key=lambda s: s["sec_id"]):
            m = SEMESTER_LINE_RE.match(sec["heading"])
            cur = int(m.group(1) or ORDINAL_WORDS.get((m.group(2) or "").lower(), 0) or (m.group(2) or 0)) if m else None
            if cur:
                found.add(cur)
            header = []
            for ln in sec["lines"]:
                ln = ln.strip()
                if ln.startswith("##SUBSECTION##"):
                    ln = ln.replace("##SUBSECTION##", "").strip()
                    if not SEMESTER_LINE_RE.match(ln):
                        continue
                if not ln:
                    continue
                sm = SEMESTER_LINE_RE.match(ln)
                if sm and "|" not in ln:
                    v = sm.group(1) or sm.group(2)
                    cur = int(v) if v.isdigit() else ORDINAL_WORDS.get(v.lower(), cur)
                    found.add(cur)
                    continue
                if "|" not in ln:
                    if len(ln.split()) >= 8 and not is_fragment(ln):
                        n = self.repair.fix(ln.replace("Note:", "").strip())
                        if n not in notes:
                            notes.append(n)
                    continue
                cells = [c.strip() for c in ln.split("|")]
                if not re.search(r"\d", ln) and re.search(r"\b(course|credits?|ltp|semester|name|code)\b", ln, re.I):
                    header = [re.sub(r"\s+", " ", c) for c in cells]
                    continue
                # tables with explicit "Sno" / "Semester" columns (positional)
                sem_idx = next((i for i, h in enumerate(header) if re.fullmatch(r"semester", h.strip(), re.I)), None)
                sno_idx = 0 if header and re.match(r"^s\.?\s*no", header[0], re.I) else None
                if header and len(cells) == len(header) and (sem_idx is not None or sno_idx is not None):
                    pos_vals = []
                    for i, c in enumerate(cells):
                        v = c
                        if ":" in c and c.split(":", 1)[0].strip() == header[i].strip():
                            v = c.split(":", 1)[1].strip()
                        elif ": " in c:            # two PDF rows merged -> newer value
                            v = c.split(": ")[-1].strip()
                        if i == sem_idx:
                            m_s = re.fullmatch(r"(\d)\s*(?:st|nd|rd|th)?", v.strip(), re.I)
                            if m_s:
                                cur = int(m_s.group(1))
                                found.add(cur)
                            continue
                        if i == sno_idx:
                            continue
                        pos_vals.append(v)
                    vals = [v for v in pos_vals if v]
                else:
                    vals = clean_table_row(ln, header)
                if vals:
                    sm = SEMESTER_LINE_RE.match(vals[0])
                    if sm:
                        v = sm.group(1) or sm.group(2)
                        cur = int(v) if v.isdigit() else ORDINAL_WORDS.get(v.lower(), cur)
                        found.add(cur)
                        vals = vals[1:]
                row = self._format_course_row(vals)
                if not row:
                    continue
                key = re.sub(r"\W+", "", row.lower())
                if key in seen:
                    continue
                seen.add(key)
                rows.append((cur, row))
        return rows, notes, found

    # -- (b) fact lines ----------------------------------------------------
    def _fact_lines(self, query, goal, ranked_sections, force=False, min_coverage=0.5, flat=False):
        if not force and not FACT_CUE_RE.search(query):
            return None
        q_toks = set(content_tokens(query))
        if re.search(r"\b(duration|how\s+long|how\s+many\s+years|length)\b", query, re.I) and \
                not re.search(r"\b(test|exam|entrance|interview)\b", query, re.I):
            q_toks = (q_toks - {"duration"}) | {"year"}
        if re.search(r"\bindian?s?\b", query, re.I):
            q_toks |= {"india"}
        q_toks -= {"how", "many", "much"}
        if not q_toks:
            return None
        total_w = sum(self.w(t) for t in q_toks)

        progs = goal.get("programs") or set()
        pool = ranked_sections
        if progs:
            owned = [(sc, s) for sc, s in ranked_sections if s.get("file_owned") and progs & s["programs"]]
            if owned:
                pool = owned
        candidates = []
        for rank, (score, sec) in enumerate(pool[:12] if flat else pool[:6]):
            heading = sec["heading"]
            items = list(enumerate(sec["lines"]))
            if len(heading.split()) >= 6:      # some PDFs store content in the heading itself
                items = [(-1, heading)] + items
            for idx, ln in items:
                if ln.startswith("##SUBSECTION##"):
                    heading = ln.replace("##SUBSECTION##", "").strip()
                    continue
                pieces = re.split(r"(?<=[.;])\s+(?=[A-Z])", ln) if len(ln) > 220 else [ln]
                for piece in pieces:
                    tok_list = content_tokens(piece)
                    toks = set(tok_list)
                    toks |= set(content_tokens(heading)) & {"seat", "test", "entrance", "eligib", "fee"}
                    hit = q_toks & toks
                    if not hit:
                        continue
                    tf = Counter(tok_list)
                    s = sum(self.w(t) * (1 + 0.25 * math.log(max(1, tf.get(t, 1)))) for t in hit)
                    s *= 1.0 if flat else (1.0 - 0.08 * rank)   # prefer better ranked sections
                    if re.search(r"\d", piece):
                        s *= 1.15                          # facts usually carry a number
                    candidates.append((s, rank, sec["sec_id"], idx, piece.strip(), heading, sec))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0], reverse=True)
        top = candidates[0]
        coverage = min(1.0, top[0] / (total_w * 1.15)) if not re.search(r"\d", top[4]) else top[0] / (total_w * 1.15)
        hit_all = q_toks <= (set(content_tokens(top[4])) | {"year"} if "year" in q_toks else set(content_tokens(top[4])))
        if min_coverage >= 0.99 and not hit_all:
            return None
        if coverage < min_coverage:
            return None
        chosen, seen = [], set()
        for c in candidates:
            if c[0] < top[0] * 0.85 or len(chosen) >= 3:
                break
            if c[2] != top[2]:          # keep the answer inside one fact/section
                continue
            key = re.sub(r"\W+", "", c[4].lower())
            if key in seen:
                continue
            seen.add(key)
            chosen.append(c)
        # drop a line whose words are all contained in another chosen line
        tok_sets = [set(content_tokens(c[4])) for c in chosen]
        chosen = [c for i, c in enumerate(chosen)
                  if not any(i != j and tok_sets[i] < tok_sets[j] for j in range(len(chosen)))]
        chosen.sort(key=lambda c: (c[1], c[2], c[3]))
        lines = [self._format_line(c[4]) for c in chosen]
        sec = top[6]
        if len(chosen) >= 1 and chosen[0][4].rstrip().endswith(":") and chosen[0][3] >= 0:
            follow = []
            for ln in sec["lines"][chosen[0][3] + 1:]:
                if re.match(r"^\s*(\d+[.)]|[a-z][.)]|[•\-*])\s+", ln):
                    follow.append(ln)
                elif follow:
                    break
            if follow:
                lines = [self._format_line(chosen[0][4])] + [self._format_line(f) for f in follow[:12]]
        heading = top[5] if good_heading(top[5]) else (sec["heading"] if good_heading(sec["heading"])
                                                       else ("Course Details" if "|" in top[4] else sec["doc_title"]))
        return heading, lines, [sec]

    # -- (d) best block ----------------------------------------------------
    def _best_block(self, query, goal, sec, label):
        q_toks = set(content_tokens(query))
        progs = goal.get("programs") or set()
        blocks = split_blocks(sec)
        # merge "empty" blocks (headings without body) into the following one
        merged = []
        for h, body in blocks:
            if merged and not merged[-1][1] and len(merged[-1][0].split()) <= 3:
                merged[-1] = [merged[-1][0] + " " + h, body]
            else:
                merged.append([h, body])
        blocks = [b for b in merged if b[1] or b is merged[0]]

        scored = []
        for i, (h, body) in enumerate(blocks):
            h_t = set(content_tokens(h))
            b_t = Counter(t for ln in body for t in content_tokens(ln))
            s = 0.0
            for t in q_toks:
                if t in h_t:
                    s += 3.0 * self.w(t)
                if t in b_t:
                    s += self.w(t) * (1 + math.log(b_t[t]))
            if i == 0:
                s += 0.5  # the section's own heading is the default answer
            scored.append((s, i))
        scored.sort(reverse=True)
        bi = scored[0][1]
        head, body = blocks[bi]
        body = list(body)

        # a block that ends with ':' continues in the next block / next section
        used = [sec]
        if body and body[-1].rstrip().endswith(":") or (not body):
            nxt = None
            for h2, b2 in blocks[bi + 1:]:
                if progs and self.compiler.detect_explicit_programs(h2) & progs:
                    nxt = (h2, b2)
                    break
            if nxt is None and bi + 1 < len(blocks):
                nxt = tuple(blocks[bi + 1])
            if nxt is None:
                following = [s for s in self.kb["sections"] if s["source_file"] == sec["source_file"]
                             and sec["sec_id"] < s["sec_id"] <= sec["sec_id"] + 4]
                pick = None
                for s2 in following:
                    if progs and self.compiler.detect_explicit_programs(s2["heading"]) & progs:
                        pick = s2
                        break
                pick = pick or (following[0] if following else None)
                if pick:
                    nxt = (pick["heading"], [l for l in pick["lines"] if not l.startswith("##SUBSECTION##")])
                    used.append(pick)
            if nxt:
                body += ["##SUBSECTION## " + nxt[0]] + list(nxt[1])

        # pull in a directly referenced follow-up block (e.g. 'closely related discipline*' -> list)
        if bi + 1 < len(blocks):
            h2, b2 = blocks[bi + 1]
            h2_t = set(content_tokens(h2))
            body_t = set(t for ln in body for t in content_tokens(ln))
            if len(h2_t & body_t) >= 2 and len(h2_t & body_t) / max(1, len(h2_t)) >= 0.5:
                body += ["##SUBSECTION## " + h2] + b2

        meaningful = [l for l in body if not l.startswith("##SUBSECTION##") and
                      (len(l.split()) >= 4 or re.search(r"\w+\s*:\s*\S+", l))]
        if not meaningful:
            return NOT_FOUND, used
        title = head if bi > 0 else sec["heading"]
        return self._render_block(label, title, body), used

    def _render_block(self, label, title, lines):
        title = self.repair.fix(title.strip(" -—:"))
        out = [f"**{label + ' — ' if label and label.lower() not in title.lower() else ''}{title}**"]
        seen = {re.sub(r"\W+", "", title.lower())}
        for ln in lines:
            clean = ln.strip()
            if not clean:
                continue
            key = re.sub(r"\W+", "", clean.replace("##SUBSECTION##", "").lower())
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)
            out.append(self._format_line(clean))
        return "\n".join(out)

    def _format_line(self, line):
        clean = self.repair.fix(line.strip())
        if clean.startswith("##SUBSECTION##"):
            return f"\n**{clean.replace('##SUBSECTION##', '').strip().rstrip(':')}**"
        clean = clean.lstrip("•-* ").strip()
        if "|" in clean:
            if re.search(r"\b\d \d \d\b", clean):           # looks like a course row
                row = self._format_course_row(clean_table_row(clean, []))
                if row:
                    return row
            parts = [p.strip() for p in clean.split("|") if p.strip()]
            return "• " + " — ".join(parts)
        if ":" in clean and len(clean) < 140:
            k, v = clean.split(":", 1)
            if len(k.split()) <= 6 and v.strip():
                return f"• **{k.strip()}:** {v.strip()}"
        return f"• {clean}"
