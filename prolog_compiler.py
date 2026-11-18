import os
import re
import math
import pickle
import fitz  # PyMuPDF
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class PDFToPrologCompiler:
    def __init__(self, pdf_dir="AI_Knowledge_base", pl_file="department_kb.pl", pkl_file="prolog_kb.pkl"):
        self.pdf_dir = pdf_dir if os.path.isabs(pdf_dir) else os.path.join(BASE_DIR, pdf_dir)
        if not os.path.exists(self.pdf_dir) and os.path.exists(pdf_dir):
            self.pdf_dir = os.path.abspath(pdf_dir)

        self.pl_file = pl_file if os.path.isabs(pl_file) else os.path.join(BASE_DIR, pl_file)
        self.pkl_file = pkl_file if os.path.isabs(pkl_file) else os.path.join(BASE_DIR, pkl_file)

        self.stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "can", "could", "may", "might", "must", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "as", "into", "through",
            "after", "over", "between", "out", "against", "during", "without",
            "before", "under", "around", "among", "and", "but", "or", "yet",
            "so", "if", "because", "while", "please", "tell", "me", "give",
            "want", "know", "you", "what", "which", "this", "that", "these", "those",
            "about", "details", "information", "explain", "describe", "who", "how"
        }

        # Strict Program Detection Patterns (Ordered from specific to general)
        self.program_patterns = [
            ("dual_degree", r"\b(dual\s+degree|integrated\s+b\.?\s*tech|b\.?\s*tech\s*(?:and|&|\+|-)\s*m\.?\s*tech)\b"),
            ("btech_mnc", r"\b(m\s*&\s*c|mnc|math(?:s|ematics)?\s+(?:and|&)\s+computing)\b"),
            ("mca", r"\b(m\.?\s*c\.?\s*a\.?|master\s+of\s+computer\s+applications?)\b"),
            ("btech_cse", r"\b(b\.?\s*tech\.?|bachelor\s+of\s+technology)\b"),
            ("mtech", r"\b(m\.?\s*tech\.?|master\s+of\s+technology)\b"),
            ("phd", r"\b(ph\.?\s*d\.?|doctoral|doctorate)\b"),
            ("msc", r"\b(m\.?\s*sc\.?|master\s+of\s+science)\b"),
            ("bsc", r"\b(b\.?\s*sc\.?|bachelor\s+of\s+science)\b")
        ]

        # Priority-Ordered Topic Classification Rules (course_structure & eligibility checked BEFORE overview!)
        self.topic_patterns = [
            ("course_structure", r"\b(course\s+structure|program(?:me)?\s+structure|curriculum|syllabus|semesters?\s*[-:]?\s*(?:[i1-8vxl]+|\d+)|sem\s*[-:]?\s*(?:[i1-8vxl]+|\d+)|core\s+courses?|elective\s+courses?|electives?|credit\s+structure|minimum\s+credits?|scheme\s+of\s+teaching|study\s+scheme|list\s+of\s+courses|course\s+code|dissertation|project\s+work)\b"),
            ("eligibility", r"\b(eligib(?:le|ility)|admissions?\s+criteria|modes?\s+of\s+admission|admission\s+process|minimum\s+qualifications?|selection\s+process|entrance\s+test|intake|seats?\s+matrix|prerequisites?)\b"),
            ("fees", r"\b(fees?\s+structure|tuition\s+fees?|semester\s+fees?|program(?:me)?\s+fees?|fee\s+details|fees|tuition|cost)\b"),
            ("faculty", r"\b(faculty|professors?|academic\s+staff|head\s+of\s+department|h\.?o\.?d\.?)\b"),
            ("contact", r"\b(contact|administration|office\s+address|helpdesk|reach\s+us|secretariat|phone|email)\b"),
            ("overview", r"\b(overview|vision|mission|history|introduction|course\s+objectives?|program(?:me)?\s+objectives?|objectives?|facilities|laboratories|labs?|career|placements?|research\s+areas?|related\s+works?)\b")
        ]

    def detect_explicit_programs(self, text):
        low = text.lower()
        found = set()
        for prog_atom, pat in self.program_patterns:
            if re.search(pat, low, re.IGNORECASE):
                if prog_atom == "btech_cse" and ("dual_degree" in found or "btech_mnc" in found):
                    continue
                if prog_atom == "mtech" and "dual_degree" in found:
                    continue
                found.add(prog_atom)
        return found

    def classify_topic(self, text):
        low = text.lower().strip()
        for topic, pat in self.topic_patterns:
            if re.search(pat, low, re.IGNORECASE):
                return topic
        return None

    def tokenize(self, text):
        low = text.lower()
        tokens = re.findall(r'\b[a-z0-9]{2,}\b', low)
        return [self._stem(t) for t in tokens if t not in self.stopwords]

    def _stem(self, word):
        if len(word) <= 4 or word.isdigit():
            return word
        for suf in ["ibilities", "ibility", "ization", "ements", "ement", "ings", "ing", "ies", "es", "s"]:
            if word.endswith(suf) and len(word) - len(suf) >= 3:
                return (word[:-3] + "y") if suf == "ies" else word[:-len(suf)]
        return word

    def _is_faculty_header(self, line):
        clean = line.strip().lstrip("•-* ")
        if len(clean) > 80 or len(clean) < 5:
            return False
        if re.match(r'^(?:\d+[\.\)]\s*)?(?:prof\.?|professor|dr\.?|mr\.?|ms\.?|mrs\.?)\s+[A-Z][a-zA-Z\.\s\-]{3,55}$', clean, re.IGNORECASE):
            low_words = set(re.findall(r'\b[a-z]+\b', clean.lower()))
            if not (low_words & {"is", "are", "was", "has", "have", "received", "joined", "working", "completed", "teaches", "department"}):
                return True
        return False

    def _is_child_subitem(self, line):
        clean = line.strip()
        if re.match(r'^(?:mode|option|category|step|note|phase|type|part)\s*[-:]?\s*(?:\d+|[i1-5v]+|[a-d])\b', clean, re.IGNORECASE):
            return True
        if re.match(r'^(?:\(\d+\)|\([a-zA-Z]\)|\([ivx]+\)|\d+[\.\)])\s+', clean) and not re.search(r'\b(semester|eligibility|curriculum|course\s+structure|fee)\b', clean, re.IGNORECASE):
            return True
        return False

    def _is_section_boundary(self, line, is_bold=False, font_size=11.0, median_size=11.0):
        """
        Recognizes major section headings up to 14 words long (such as
        'Course Structure of M.Tech. (Computer Science & Engineering)').
        """
        clean = line.strip().lstrip("•-* ").strip()
        if not clean or len(clean) > 110 or len(clean) < 3:
            return False
        if "|" in clean:
            return False
        if self._is_child_subitem(clean):
            return False

        # A full sentence with verbs is never a section heading
        low_words = set(re.findall(r'\b[a-z]+\b', clean.lower()))
        if low_words & {"is", "are", "was", "were", "offered", "must", "should", "will", "can", "have", "has", "through", "candidates", "students", "following", "below", "various", "who", "which", "aims", "prepares", "follows", "equipping"}:
            return False

        # 1. Explicit Topic Heading (up to 14 words, e.g., "Course Structure of M.Tech. (Computer Science & Engineering)")
        if self.classify_topic(clean) and len(clean.split()) <= 14 and not clean.endswith("."):
            return True

        # 2. Bold standalone title line (up to 10 words, not ending with a period)
        if is_bold and 1 <= len(clean.split()) <= 10 and not clean.endswith((".", ",", ";")):
            return True

        # 3. Larger font title
        if font_size >= median_size + 1.0 and 1 <= len(clean.split()) <= 12 and not clean.endswith("."):
            return True

        # 4. ALL CAPS heading
        if clean.isupper() and 1 <= len(clean.split()) <= 10 and not any(c.isdigit() for c in clean):
            return True

        return False

    def _extract_page_elements(self, page):
        """
        Extracts all text blocks and tables in exact top-to-bottom vertical Y order
        so tables appear right below their heading (not at the end of the page!).
        """
        combined_elements = []

        # 1. Extract tables with their vertical Y position (bbox[1])
        table_bboxes = []
        try:
            tables = page.find_tables()
            for tab in tables:
                table_bboxes.append(tab.bbox)
                rows = tab.extract()
                if not rows:
                    continue
                headers = []
                y_pos = tab.bbox[1]
                for r_idx, row in enumerate(rows):
                    cells = [re.sub(r'\s+', ' ', str(c or '')).strip() for c in row]
                    if not any(cells):
                        continue
                    if r_idx == 0:
                        headers = [c for c in cells if c]
                        if headers:
                            combined_elements.append((y_pos + r_idx * 0.1, 0, " | ".join(headers), True, 11.0, True))
                    else:
                        non_empty = [c for c in cells if c]
                        if headers and len(headers) == len(cells):
                            paired = [f"{headers[i]}: {cells[i]}" for i in range(len(cells)) if cells[i]]
                            combined_elements.append((y_pos + r_idx * 0.1, 0, " | ".join(paired), False, 10.5, True))
                        elif non_empty:
                            combined_elements.append((y_pos + r_idx * 0.1, 0, " | ".join(non_empty), False, 10.5, True))
        except Exception:
            pass

        def _inside_any_table(bbox):
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            for tb in table_bboxes:
                if (tb[0] - 4 <= cx <= tb[2] + 4) and (tb[1] - 4 <= cy <= tb[3] + 4):
                    return True
            return False

        # 2. Extract text lines with font size & bold status, skipping duplicate cells already extracted by find_tables()
        page_dict = page.get_text("dict")
        all_sizes = []
        text_lines = []

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                l_bbox = line["bbox"]
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = "".join(s.get("text", "") for s in spans)
                line_text = re.sub(r'[ \t]+', ' ', line_text).strip()
                if len(line_text) < 2:
                    continue

                max_size = max((s.get("size", 11.0) for s in spans), default=11.0)
                is_bold = any("bold" in s.get("font", "").lower() or (s.get("flags", 0) & 16) for s in spans)
                all_sizes.append(max_size)

                # If this line is inside a detected table, only skip it if it's NOT a major section heading
                if _inside_any_table(l_bbox) and not self._is_section_boundary(line_text, is_bold, max_size, 11.0):
                    continue

                text_lines.append((round(l_bbox[1] / 3.0) * 3.0, l_bbox[0], line_text, is_bold, max_size, False))

        median_size = sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 11.0

        # Merge wrapped lines on the same paragraph before combining with table rows
        text_lines.sort(key=lambda x: (x[0], x[1]))
        merged_text_lines = []
        for y_key, x0, text, is_bold, sz, is_tab in text_lines:
            if (
                merged_text_lines
                and not self._is_section_boundary(text, is_bold, sz, median_size)
                and not self._is_faculty_header(text)
                and not self._is_child_subitem(text)
                and not text.startswith(("•", "-", "*"))
                and not merged_text_lines[-1][2].endswith((".", ":", ";", "?"))
                and not self._is_section_boundary(merged_text_lines[-1][2], merged_text_lines[-1][3], merged_text_lines[-1][4], median_size)
                and abs(sz - merged_text_lines[-1][4]) < 1.2
                and len(merged_text_lines[-1][2]) > 35
            ):
                prev = merged_text_lines[-1]
                merged_text_lines[-1] = (prev[0], prev[1], prev[2] + " " + text, prev[3], prev[4], False)
            else:
                merged_text_lines.append((y_key, x0, text, is_bold, sz, is_tab))

        all_page_items = combined_elements + merged_text_lines
        all_page_items.sort(key=lambda x: (x[0], x[1]))

        final_lines = [(item[2], item[3], item[4]) for item in all_page_items]

        if not final_lines:
            for ln in page.get_text("text", sort=True).split("\n"):
                clean_ln = re.sub(r'[ \t]+', ' ', ln).strip()
                if len(clean_ln) >= 2:
                    final_lines.append((clean_ln, False, 11.0))

        return final_lines, median_size

    def compile_knowledge_base(self):
        if not os.path.exists(self.pdf_dir):
            os.makedirs(self.pdf_dir)
            return {"sections": [], "faculty": [], "available_programs": set(), "idf": {}}

        prolog_sections = []
        prolog_faculty = []
        sec_id = 0
        fac_id = 0

        for root, _, files in os.walk(self.pdf_dir):
            rel_folder = os.path.relpath(root, self.pdf_dir)
            folder_label = "General" if rel_folder == "." else rel_folder.replace("\\", " / ").replace("_", " ")

            for fname in sorted(files):
                if not fname.lower().endswith(".pdf"):
                    continue

                fpath = os.path.join(root, fname)
                doc_title = os.path.splitext(fname)[0].replace("_", " ").replace("-", " ").strip()
                folder_and_file = f"{folder_label} {doc_title}"

                file_programs = self.detect_explicit_programs(folder_and_file)
                file_default_topic = self.classify_topic(folder_and_file) or "overview"
                is_faculty_doc = "faculty" in folder_and_file.lower()

                print(f"[PROLOG COMPILER] [{folder_label}] / {fname} -> Program Lock: {file_programs or 'Multi/General'}")

                try:
                    doc = fitz.open(fpath)
                except Exception as e:
                    print(f"[ERROR] Could not open {fname}: {e}")
                    continue

                all_doc_elements = []
                doc_median_size = 11.0
                for page_num, page in enumerate(doc, start=1):
                    elements, med_sz = self._extract_page_elements(page)
                    doc_median_size = med_sz
                    for text, is_bold, sz in elements:
                        all_doc_elements.append((page_num, text, is_bold, sz))

                if not all_doc_elements:
                    continue

                if is_faculty_doc:
                    fac_blocks = self._extract_faculty_facts(all_doc_elements, fname, folder_label, fac_id)
                    prolog_faculty.extend(fac_blocks)
                    fac_id += len(fac_blocks)

                active_programs = set(file_programs) if file_programs else {"general"}
                active_topic = file_default_topic
                current_heading = doc_title
                current_page = 1
                current_lines = []

                for page_num, text, is_bold, sz in all_doc_elements:
                    if self._is_section_boundary(text, is_bold, sz, doc_median_size):
                        new_heading = text.strip(": ").lstrip("•-* ").strip()
                        new_topic = self.classify_topic(new_heading)
                        heading_progs = self.detect_explicit_programs(new_heading)

                        # CRITICAL PARENT-CHILD GROUPING RULE:
                        # 1. If we are currently inside 'course_structure' or 'eligibility', and the new heading
                        #    is ALSO 'course_structure' (like 'Semester I', 'Core Courses', 'Electives') or
                        #    'eligibility' (like 'Modes of Admission') or a neutral sub-header (new_topic is None,
                        #    like 'Table 1' or 'Note'), keep it INSIDE the parent section as a ##SUBSECTION##!
                        is_child_of_active_topic = (
                            current_lines
                            and active_topic in {"course_structure", "eligibility"}
                            and (new_topic == active_topic or new_topic is None)
                            and (not heading_progs or heading_progs == active_programs)
                        )

                        if is_child_of_active_topic:
                            current_lines.append(f"##SUBSECTION## {new_heading}")
                        else:
                            # Close previous section before starting the new major section!
                            if current_lines:
                                prolog_sections.append(self._build_section_fact(
                                    sec_id, fname, folder_label, doc_title, current_heading,
                                    current_page, current_lines, active_programs, active_topic, bool(file_programs)
                                ))
                                sec_id += 1
                                current_lines = []

                            current_heading = new_heading
                            current_page = page_num
                            active_topic = new_topic if new_topic else "overview"
                            if not file_programs and heading_progs:
                                active_programs = heading_progs
                    else:
                        current_lines.append(text)

                if current_lines:
                    prolog_sections.append(self._build_section_fact(
                        sec_id, fname, folder_label, doc_title, current_heading,
                        current_page, current_lines, active_programs, active_topic, bool(file_programs)
                    ))
                    sec_id += 1

        all_token_lists = [s["tokens"] for s in prolog_sections] + [f["tokens"] for f in prolog_faculty]
        N = max(1, len(all_token_lists))
        df = Counter()
        for toks in all_token_lists:
            for t in set(toks):
                df[t] += 1
        idf = {term: math.log(1.0 + (N - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

        available_programs = set()
        for s in prolog_sections:
            available_programs |= (s["programs"] - {"general"})

        kb_data = {
            "sections": prolog_sections,
            "faculty": prolog_faculty,
            "available_programs": available_programs,
            "idf": idf
        }

        self._write_prolog_pl_file(prolog_sections, prolog_faculty)
        with open(self.pkl_file, "wb") as f:
            pickle.dump(kb_data, f)

        print(f"[PROLOG COMPILER] Done! {len(prolog_sections)} Prolog section facts & {len(prolog_faculty)} faculty facts.")
        return kb_data

    def _extract_faculty_facts(self, all_doc_elements, fname, folder_label, start_id):
        faculty_facts = []
        current_name = None
        current_page = 1
        current_lines = []

        for page_num, text, is_bold, _ in all_doc_elements:
            if self._is_faculty_header(text):
                if current_name and current_lines:
                    faculty_facts.append(self._build_faculty_obj(
                        start_id + len(faculty_facts), current_name, current_lines, fname, folder_label, current_page
                    ))
                current_name = text.strip().lstrip("•-* ")
                current_page = page_num
                current_lines = [current_name]
            else:
                if current_name is not None:
                    if is_bold and self.classify_topic(text) in {"course_structure", "fees", "eligibility", "contact"}:
                        faculty_facts.append(self._build_faculty_obj(
                            start_id + len(faculty_facts), current_name, current_lines, fname, folder_label, current_page
                        ))
                        current_name = None
                        current_lines = []
                    else:
                        current_lines.append(text)

        if current_name and current_lines:
            faculty_facts.append(self._build_faculty_obj(
                start_id + len(faculty_facts), current_name, current_lines, fname, folder_label, current_page
            ))
        return faculty_facts

    def _build_faculty_obj(self, fac_id, full_name, lines, fname, folder_label, page):
        raw_words = re.findall(r'\b[a-z]{2,}\b', full_name.lower())
        ignore_titles = {"prof", "professor", "dr", "mr", "ms", "mrs", "associate", "assistant", "head", "department", "chairperson", "shri", "smt"}
        name_tokens = {w for w in raw_words if w not in ignore_titles}
        person_atom = "_".join(w for w in raw_words if w not in ignore_titles) or f"faculty_{fac_id}"
        full_text = "\n".join(lines)

        return {
            "fac_id": fac_id,
            "person_atom": person_atom,
            "name_tokens": name_tokens,
            "full_name": full_name,
            "lines": lines,
            "full_text": full_text,
            "source_file": fname,
            "folder": folder_label,
            "page": page,
            "tokens": self.tokenize(f"faculty professor {full_name} {full_text}")
        }

    def _build_section_fact(self, sec_id, fname, folder, doc_title, heading, page, lines, active_programs, active_topic, file_owned):
        clean_body = "\n".join(lines).replace("##SUBSECTION##", "").strip()
        programs = set(active_programs)

        if not file_owned and programs == {"general"}:
            detected = self.detect_explicit_programs(f"{heading} {clean_body}")
            if len(detected) == 1:
                programs = detected

        topic = self.classify_topic(heading) or active_topic
        if topic == "overview":
            inferred = self.classify_topic(clean_body[:350])
            if inferred in {"course_structure", "eligibility", "fees"}:
                topic = inferred

        search_blob = f"{folder} {doc_title} {heading} {' '.join(programs)} {topic} {clean_body}"
        return {
            "sec_id": sec_id,
            "source_file": fname,
            "folder": folder,
            "doc_title": doc_title,
            "heading": heading,
            "page": page,
            "lines": lines,
            "full_text": clean_body,
            "programs": programs,
            "file_owned": file_owned,
            "topic": topic,
            "tokens": self.tokenize(search_blob),
            "heading_tokens": self.tokenize(heading),
            "body_tokens": self.tokenize(clean_body)
        }

    def _write_prolog_pl_file(self, sections, faculty):
        with open(self.pl_file, "w", encoding="utf-8") as f:
            f.write("% =====================================================================\n")
            f.write("% PROLOG KNOWLEDGE BASE (department_kb.pl)\n")
            f.write("% =====================================================================\n")
            f.write(":- discontiguous faculty_fact/5.\n")
            f.write(":- discontiguous section_fact/7.\n\n")

            for fac in faculty:
                safe_name = fac["full_name"].replace('"', '\\"')
                safe_text = fac["full_text"].replace('"', '\\"').replace("\n", "\\n")
                f.write(f'faculty_fact({fac["person_atom"]}, "{safe_name}", "{fac["source_file"]}", {fac["page"]}, "{safe_text}").\n')

            f.write("\n")
            for sec in sections:
                safe_head = sec["heading"].replace('"', '\\"')
                safe_text = sec["full_text"].replace('"', '\\"').replace("\n", "\\n")
                for prog in sorted(sec["programs"]):
                    f.write(
                        f'section_fact({sec["sec_id"]}, {prog}, {sec["topic"]}, '
                        f'"{safe_head}", "{sec["source_file"]}", {sec["page"]}, "{safe_text}").\n'
                    )

            f.write("\n% --- Core Prolog Unification Rules ---\n")
            f.write("unify_faculty(Person, Name, File, Page, Profile) :-\n")
            f.write("    faculty_fact(Person, Name, File, Page, Profile).\n\n")
            f.write("unify_section(Program, Topic, Heading, File, Page, Content) :-\n")
            f.write("    section_fact(_, Program, Topic, Heading, File, Page, Content).\n")

if __name__ == "__main__":
    PDFToPrologCompiler().compile_knowledge_base()