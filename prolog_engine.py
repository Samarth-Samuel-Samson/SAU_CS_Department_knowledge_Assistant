import os
import re
import math
import pickle
from collections import Counter
from prolog_compiler import PDFToPrologCompiler
from precise_answers import PrecisionLayer

class PrologInferenceEngine:
    def __init__(self, pdf_dir="AI_Knowledge_base", pl_file="department_kb.pl", pkl_file="prolog_kb.pkl"):
        self.compiler = PDFToPrologCompiler(pdf_dir=pdf_dir, pl_file=pl_file, pkl_file=pkl_file)
        self.pkl_file = self.compiler.pkl_file
        self.kb = None
        self.precision = None
        self.load_or_compile()

    def load_or_compile(self, force_rebuild=False):
        if force_rebuild or not os.path.exists(self.pkl_file):
            self.kb = self.compiler.compile_knowledge_base()
        else:
            try:
                with open(self.pkl_file, "rb") as f:
                    self.kb = pickle.load(f)
            except Exception:
                self.kb = None

            if not self.kb or not self.kb.get("sections") or "heading_tokens" not in self.kb["sections"][0]:
                self.kb = self.compiler.compile_knowledge_base()

        # Precision layer: reads the KB (never modifies it) to return exact answers
        self.precision = PrecisionLayer(self.kb, self.compiler, self.compiler.pdf_dir)

    def _parse_prolog_goal(self, raw_query):
        q_lower = raw_query.lower()
        q_tokens = self.compiler.tokenize(raw_query)
        q_set = set(q_tokens)
        q_words_raw = set(re.findall(r'\b[a-z0-9]+\b', q_lower))

        # 1. Unify Program Atom (mca, btech_cse, btech_mnc, dual_degree, mtech, phd)
        programs = self.compiler.detect_explicit_programs(raw_query)

        # 2. Unify Topic Atom (Priority-filtered so 'overview' is never added if a specific topic is asked!)
        topics = set()
        for topic_name, pat in self.compiler.topic_patterns:
            if re.search(pat, q_lower, re.IGNORECASE):
                topics.add(topic_name)

        if q_set & {"eligib", "eligibility", "admission", "qualify", "qualification", "criteria", "entrance", "intake", "seat", "mode"}:
            topics.add("eligibility")
        if q_set & {"structure", "curriculum", "syllabus", "semester", "credit", "elective", "subject", "cours"}:
            topics.add("course_structure")
        if q_set & {"fee", "fees", "tuition", "cost", "charge", "dollar", "usd", "inr", "rupee"}:
            topics.add("fees")
        if q_set & {"faculty", "professor", "teacher", "hod", "staff"}:
            topics.add("faculty")
        if q_set & {"contact", "administration", "admin", "office", "email", "phone", "address", "reach"}:
            topics.add("contact")

        # CRITICAL FIX: If a specific topic (course_structure, eligibility, fees, faculty, contact)
        # is present in the query, remove 'overview' so it never matches Course Objectives by mistake!
        specific_topics = topics - {"overview"}
        if specific_topics:
            topics = specific_topics

        # 3. Unify Faculty Person Atom
        matched_faculty = []
        for fac in self.kb.get("faculty", []):
            overlap = fac["name_tokens"] & (q_words_raw | q_set)
            if overlap:
                ratio = len(overlap) / max(1, len(fac["name_tokens"]))
                matched_faculty.append((len(overlap), ratio, fac))
        matched_faculty.sort(key=lambda x: (x[0], x[1]), reverse=True)
        target_faculty = matched_faculty[0][2] if matched_faculty else None

        return {
            "raw_query": raw_query,
            "tokens": q_tokens,
            "programs": programs,
            "topics": topics,
            "target_faculty": target_faculty
        }
        q_lower = raw_query.lower()
        q_tokens = self.compiler.tokenize(raw_query)
        q_set = set(q_tokens)
        q_words_raw = set(re.findall(r'\b[a-z0-9]+\b', q_lower))

        # 1. Unify Program Atom (mca, btech_cse, btech_mnc, dual_degree, mtech, phd)
        programs = self.compiler.detect_explicit_programs(raw_query)

        # 2. Unify Topic Atom (eligibility, course_structure, fees, faculty, contact, overview)
        topics = set()
        for topic_name, pat in self.compiler.topic_patterns.items():
            if re.search(pat, q_lower, re.IGNORECASE):
                topics.add(topic_name)

        if q_set & {"eligib", "eligibility", "admission", "qualify", "qualification", "criteria", "entrance", "intake", "seat", "mode"}:
            topics.add("eligibility")
        if q_set & {"structure", "curriculum", "syllabus", "semester", "credit", "elective", "subject"}:
            topics.add("course_structure")
        if q_set & {"fee", "fees", "tuition", "cost", "charge", "dollar", "usd", "inr", "rupee"}:
            topics.add("fees")
        if q_set & {"faculty", "professor", "teacher", "hod", "staff"}:
            topics.add("faculty")
        if q_set & {"contact", "administration", "admin", "office", "email", "phone", "address", "reach"}:
            topics.add("contact")

        # 3. Unify Faculty Person Atom
        matched_faculty = []
        for fac in self.kb.get("faculty", []):
            overlap = fac["name_tokens"] & (q_words_raw | q_set)
            if overlap:
                ratio = len(overlap) / max(1, len(fac["name_tokens"]))
                matched_faculty.append((len(overlap), ratio, fac))
        matched_faculty.sort(key=lambda x: (x[0], x[1]), reverse=True)
        target_faculty = matched_faculty[0][2] if matched_faculty else None

        return {
            "raw_query": raw_query,
            "tokens": q_tokens,
            "programs": programs,
            "topics": topics,
            "target_faculty": target_faculty
        }

    def _valid_line_for_program(self, line, asked_progs, asked_topics):
        """
        PROLOG GUARD RULE:
        Blocks any line that talks about a different degree program (e.g. B.Tech inside an MCA query),
        UNLESS the line is describing qualifying undergraduate degrees for admission/eligibility!
        """
        if not asked_progs:
            return True

        line_progs = self.compiler.detect_explicit_programs(line)
        if not line_progs or (line_progs & asked_progs):
            return True

        # If the line mentions another degree (e.g. B.Sc. / B.Tech), only allow it if it's an eligibility qualification statement
        if "eligibility" in asked_topics:
            low = line.lower()
            if any(w in low for w in ["bachelor", "graduate", "qualifying", "minimum", "marks", "50%", "55%", "60%", "10+2", "passed", "obtained", "recognized", "equivalent"]):
                return True

        # Otherwise, it belongs to a different course -> reject!
        return False

    def run_rag(self, user_query, top_k=5):
        response = self._run_rag_inner(user_query, top_k)
        if self.precision:
            try:
                response = self.precision.final_filter(user_query, response)
            except Exception as exc:
                print(f"[PRECISION] final filter failed: {exc}")
            trace = response.get("rag_trace")
            if isinstance(trace, dict):
                trace["precision_version"] = __import__("precise_answers").PRECISION_VERSION
        return response

    def _run_rag_inner(self, user_query, top_k=5):
        fallback = "Information not available in the department knowledge base. Please contact the Computer Science Department Administration."

        if not self.kb or not self.kb.get("sections"):
            self.load_or_compile(force_rebuild=True)

        if not self.kb or not self.kb.get("sections") or not user_query or len(user_query.strip()) < 2:
            return {"found": False, "answer": fallback, "confidence": 0.0, "sources": [], "rag_trace": {}}

        # PRECISION: fix spelling mistakes using the knowledge base's own vocabulary
        if self.precision:
            user_query = self.precision.correct_spelling(user_query)

        goal = self._parse_prolog_goal(user_query)

        # PRECISION RULE 0: exact answers for faculty members, roles, department office and fees
        direct = self.precision.try_direct(user_query, goal) if self.precision else None
        if direct:
            return direct

        if not goal["tokens"] and not goal["target_faculty"]:
            return {"found": False, "answer": fallback, "confidence": 0.0, "sources": [], "rag_trace": {}}

        # =========================================================================
        # PROLOG RULE 1: Faculty Unification
        # ?- unify_faculty(PersonAtom, Name, File, Page, Profile).
        # =========================================================================
        if goal["target_faculty"] and "fees" not in goal["topics"] and "course_structure" not in goal["topics"]:
            fac = goal["target_faculty"]
            formatted_profile = self._format_faculty_profile(fac)
            return {
                "found": True,
                "answer": formatted_profile,
                "confidence": 99.4,
                "sources": [{
                    "file": fac["source_file"],
                    "category": fac["folder"],
                    "page": fac["page"],
                    "section": fac["full_name"]
                }],
                "rag_trace": {
                    "expanded_query": f"?- unify_faculty({fac['person_atom']}, Name, File, Page, Profile).",
                    "status": f"Prolog Unification Succeeded: Bound to {fac['full_name']}",
                    "retrieved_chunks": [{
                        "chunk_id": fac["fac_id"],
                        "file": fac["source_file"],
                        "category": fac["folder"],
                        "section": fac["full_name"],
                        "page": fac["page"],
                        "score": 1.0,
                        "snippet": fac["full_text"][:260]
                    }]
                }
            }

        asked_progs = goal["programs"]
        avail_progs = self.kb.get("available_programs", set())
        # B.Tech (Mathematics & Computing) facts are stored under btech_cse (from its own PDF)
        if "btech_mnc" in asked_progs and "btech_mnc" not in avail_progs:
            asked_progs = (asked_progs - {"btech_mnc"}) | {"btech_cse"}
            goal["programs"] = asked_progs
            goal["prefer_file"] = "math"
        if asked_progs and avail_progs and not (asked_progs & avail_progs):
            return {
                "found": False,
                "answer": fallback,
                "confidence": 0.0,
                "sources": [],
                "rag_trace": {
                    "expanded_query": f"?- unify_section({list(asked_progs)[0]}, _, _, _, _, _). -> false.",
                    "status": f"Prolog Unification Failed: Course {list(asked_progs)} not in KB.",
                    "retrieved_chunks": []
                }
            }

        # =========================================================================
        # PROLOG RULE 2: Exact Course Fee Unification
        # ?- unify_section(Program, fees, Heading, File, Page, Content).
        # =========================================================================
        if "fees" in goal["topics"]:
            fee_res = self._solve_fee_goal(goal)
            if fee_res:
                return fee_res

        # =========================================================================
        # PROLOG RULE 3: Strict Program + Topic Section Unification
        # ?- unify_section(Program, Topic, Heading, File, Page, Content).
        # =========================================================================
        return self._solve_section_goal(goal, fallback)

    def _solve_fee_goal(self, goal):
        asked_progs = goal["programs"]
        q_set = set(goal["tokens"])
        prog_label = ", ".join(p.upper().replace("_", " ") for p in asked_progs) if asked_progs else "the program"

        candidate_lines = []
        seen_lines = set()

        for sec in self.kb["sections"]:
            if asked_progs and sec["programs"] != {"general"} and not (asked_progs & sec["programs"]):
                continue

            sec_is_fee = (sec["topic"] == "fees") or ("fee" in sec["folder"].lower()) or ("fee" in sec["doc_title"].lower())
            if not sec_is_fee and not re.search(r'(\$|usd|dollar|inr|rs\.?|₹|tuition|fee)', sec["full_text"].lower()):
                continue

            for idx, ln in enumerate(sec["lines"]):
                clean = ln.replace("##SUBSECTION##", "").strip()
                if len(clean) < 3:
                    continue

                if not self._valid_line_for_program(clean, asked_progs, {"fees"}):
                    continue

                norm = re.sub(r'\W+', '', clean.lower())
                if norm in seen_lines:
                    continue
                seen_lines.add(norm)

                ln_toks = set(self.compiler.tokenize(clean))
                line_progs = self.compiler.detect_explicit_programs(clean)
                score = len(q_set & ln_toks) * 1.5

                if asked_progs and ((line_progs & asked_progs) or (sec["file_owned"] and (asked_progs & sec["programs"]))):
                    score += 6.0
                if re.search(r'(\$\s?\d+|\b\d{2,6}\b)', clean.lower()):
                    score += 4.0
                if re.search(r'(dollar|usd|inr|rupee|rs\.?|₹|semester|annual|year|tuition|fee)', clean.lower()):
                    score += 2.5

                if score >= 5.0:
                    candidate_lines.append((score, idx, clean, sec))

        if not candidate_lines:
            return None

        candidate_lines.sort(key=lambda x: x[0], reverse=True)
        best_score = candidate_lines[0][0]
        top_items = [item for item in candidate_lines if item[0] >= best_score * 0.70][:4]
        top_items.sort(key=lambda x: x[1])

        best_line = candidate_lines[0][2].lstrip("•-* ").strip()
        best_sec = candidate_lines[0][3]

        amount_match = re.search(
            r'(\$\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?(?:usd|dollars?|dollors?|inr|rupees?|rs\.?)|(?:rs\.?|inr|₹)\s?\d[\d,]*(?:\.\d+)?)',
            best_line,
            re.IGNORECASE
        )
        period_match = re.search(r'(per\s+semester|per\s+year|per\s+annum|semester|annual)', best_line, re.IGNORECASE)

        if amount_match:
            amt = amount_match.group(1).strip()
            period = period_match.group(1).lower() if period_match else "per semester"
            if not period.startswith("per"):
                period = f"per {period}"
            lead = f"The fee for **{prog_label}** is **{amt} {period}**."
        else:
            lead = f"**{prog_label} Fee Structure**"

        bullets = [self._format_clean_line(item[2]) for item in top_items]
        answer = f"{lead}\n\n" + "\n".join(bullets)

        return {
            "found": True,
            "answer": answer,
            "confidence": 99.1,
            "sources": [{
                "file": best_sec["source_file"],
                "category": best_sec["folder"],
                "page": best_sec["page"],
                "section": best_sec["heading"]
            }],
            "rag_trace": {
                "expanded_query": f"?- unify_section({list(asked_progs) or '_'}, fees, Heading, File, Page, Content).",
                "status": "Prolog Fee Unification Succeeded",
                "retrieved_chunks": [{
                    "chunk_id": best_sec["sec_id"],
                    "file": best_sec["source_file"],
                    "category": best_sec["folder"],
                    "section": best_sec["heading"],
                    "page": best_sec["page"],
                    "score": round(best_score, 2),
                    "snippet": best_line
                }]
            }
        }

    def _solve_section_goal(self, goal, fallback):
        asked_progs = goal["programs"]
        asked_topics = goal["topics"]
        q_tokens = goal["tokens"]
        q_set = set(q_tokens)
        idf = self.kb.get("idf", {})

        scored_sections = []

        for sec in self.kb["sections"]:
            # 1. HARD PROLOG COURSE UNIFICATION:
            # If user asked for a specific course (e.g. MCA), NEVER unify with another course's section!
            if asked_progs:
                if sec["programs"] == {"general"}:
                    # Only allow a 'general' section if it explicitly mentions the asked program inside its text
                    if not (asked_progs & self.compiler.detect_explicit_programs(sec["full_text"])):
                        continue
                elif not (asked_progs & sec["programs"]):
                    continue

            # 2. HARD PROLOG TOPIC UNIFICATION:
            # If user asked for a specific topic (e.g. 'eligibility'), ONLY allow sections whose topic matches!
            if asked_topics and sec["topic"] not in asked_topics:
                continue

            # 3. Score matching facts by Heading match + Body IDF match
            sec_tok_counts = Counter(sec["tokens"])
            body_tok_counts = Counter(sec["body_tokens"])
            head_tok_set = set(sec["heading_tokens"])

            lex_score = 0.0
            matched_terms = 0
            for qt in q_set:
                if qt in sec_tok_counts:
                    matched_terms += 1
                    term_idf = idf.get(qt, 1.0)
                    body_tf = body_tok_counts.get(qt, 0)
                    head_bonus = 2.5 if qt in head_tok_set else 0.0
                    lex_score += term_idf * (1.5 * math.log(1 + body_tf) + head_bonus)

            if asked_topics and sec["topic"] in asked_topics:
                lex_score = (lex_score + 2.0) * 2.5
            if asked_progs and (asked_progs & sec["programs"]):
                lex_score *= 2.0
            lex_score *= self._precision_boost(goal, sec)

            if lex_score > 0:
                scored_sections.append((lex_score, matched_terms, sec))

        if not scored_sections:
            return {"found": False, "answer": fallback, "confidence": 0.0, "sources": [], "rag_trace": {}}

        scored_sections.sort(key=lambda x: x[0], reverse=True)
        best_score, best_matches, best_sec = scored_sections[0]

        if best_score < 1.2 or (best_matches == 0 and not asked_topics):
            return {"found": False, "answer": fallback, "confidence": 10.0, "sources": [], "rag_trace": {}}

        # PRECISION RULE: narrow the winning fact(s) down to exactly what was asked
        focused = None
        if self.precision:
            try:
                focused = self.precision.focus(goal["raw_query"], goal, [(sc, s) for sc, _, s in scored_sections])
            except Exception as exc:  # never fail a query because of the precision layer
                print(f"[PRECISION] focus failed: {exc}")
        if focused and focused[0] == "__NOT_FOUND__":
            return {"found": False, "answer": fallback, "confidence": 0.0, "sources": [], "rag_trace": {
                "expanded_query": f"?- unify_section({list(asked_progs) or '_'}, {list(asked_topics) or '_'}, _, _, _, _). -> false.",
                "status": "Prolog Unification Failed: no fact answers this query", "retrieved_chunks": []}}
        if focused:
            answer_text, used_secs = focused
            return {
                "found": True,
                "answer": answer_text,
                "confidence": min(99.4, round(min(1.0, best_score / 10.0) * 30 + 68.0, 1)),
                "sources": [{
                    "file": used_secs[0]["source_file"],
                    "category": used_secs[0]["folder"],
                    "page": used_secs[0]["page"],
                    "section": used_secs[0]["heading"],
                }],
                "rag_trace": {
                    "expanded_query": (
                        f"?- unify_section({list(asked_progs) or '_'}, "
                        f"{list(asked_topics) or best_sec['topic']}, Heading, \"{used_secs[0]['source_file']}\", Page, Content)."
                    ),
                    "status": f"Prolog Unification Succeeded: precise answer from {len(used_secs)} fact(s)",
                    "retrieved_chunks": [{
                        "chunk_id": s["sec_id"], "file": s["source_file"], "category": s["folder"],
                        "section": s["heading"], "page": s["page"], "score": round(best_score, 2),
                        "snippet": s["full_text"][:260] + ("..." if len(s["full_text"]) > 260 else ""),
                    } for s in used_secs],
                },
            }

        # STRICT PRECISION SELECTION:
        # For Eligibility, Overview, Contact, or Specific Queries -> Return ONLY the winning section
        # (plus any immediate sibling section in the same file that has the EXACT SAME topic and high score).
        # For Course Structure -> Allow all semester/course_structure sections from that same PDF.
        target_file = best_sec["source_file"]
        target_folder = best_sec["folder"]
        selected_secs = [best_sec]

        if "course_structure" in asked_topics:
            for sc, _, sec in scored_sections[1:]:
                if (
                    sec["source_file"] == target_file
                    and sec["folder"] == target_folder
                    and sec["topic"] == "course_structure"
                    and sc >= best_score * 0.45
                ):
                    selected_secs.append(sec)
        elif asked_topics:
            for sc, _, sec in scored_sections[1:]:
                if (
                    sec["source_file"] == target_file
                    and sec["folder"] == target_folder
                    and sec["topic"] == best_sec["topic"]
                    and abs(sec["sec_id"] - best_sec["sec_id"]) == 1
                    and sc >= best_score * 0.65
                ):
                    selected_secs.append(sec)

        selected_secs.sort(key=lambda s: s["sec_id"])
        answer_text = self._format_strict_sections(selected_secs, goal)

        if not answer_text.strip():
            return {"found": False, "answer": fallback, "confidence": 0.0, "sources": [], "rag_trace": {}}

        sources = [{
            "file": best_sec["source_file"],
            "category": best_sec["folder"],
            "page": best_sec["page"],
            "section": best_sec["heading"]
        }]

        trace_chunks = [{
            "chunk_id": s["sec_id"],
            "file": s["source_file"],
            "category": s["folder"],
            "section": s["heading"],
            "page": s["page"],
            "score": round(best_score, 2),
            "snippet": s["full_text"][:260] + ("..." if len(s["full_text"]) > 260 else "")
        } for s in selected_secs]

        prolog_goal_str = (
            f"?- unify_section({list(asked_progs) or '_'}, "
            f"{list(asked_topics) or best_sec['topic']}, Heading, \"{target_file}\", Page, Content)."
        )

        return {
            "found": True,
            "answer": answer_text,
            "confidence": min(99.4, round(min(1.0, best_score / 10.0) * 30 + 68.0, 1)),
            "sources": sources,
            "rag_trace": {
                "expanded_query": prolog_goal_str,
                "status": f"Prolog Unification Succeeded: Matched {len(selected_secs)} exact fact(s) in {target_file}",
                "retrieved_chunks": trace_chunks
            }
        }

    def _precision_boost(self, goal, sec):
        """Ranking corrections so the programme's own PDF wins over general documents."""
        q = goal["raw_query"].lower()
        asked_progs = goal["programs"]
        f = 1.0
        fname = sec["source_file"].lower()
        if asked_progs:
            if sec.get("file_owned") and (asked_progs & sec["programs"]):
                f *= 2.5
            elif sec["programs"] == {"general"}:
                f *= 0.4
        # entrance-test syllabus is not the course structure (and vice versa)
        is_entrance = bool(re.search(r"syllabus", sec["heading"], re.I) and re.search(r"entrance|test", sec["heading"], re.I))
        asks_entrance = bool(re.search(r"\b(entrance|admission\s+test|test\s+syllabus)\b", q))
        if is_entrance and asks_entrance and re.search(r"\bsyllabus\b", q):
            f *= 4.0
        if "course_structure" in goal["topics"]:
            if is_entrance and not asks_entrance:
                f *= 0.15
            elif is_entrance and asks_entrance:
                f *= 3.0
        # B.Tech CSE vs B.Tech Mathematics & Computing
        if goal.get("prefer_file"):
            f *= 3.0 if goal["prefer_file"] in fname else 0.3
        elif "btech_cse" in asked_progs and "math" in fname and not re.search(r"\bmath", q):
            f *= 0.3
        # 1-year vs 2-year M.Sc.
        if "msc" in asked_progs:
            one_year = bool(re.search(r"\b(1|one)[\s-]*year", q))
            if "1_year" in fname:
                f *= 3.0 if one_year else 0.5
        # fact-bearing tables are better than one-line fragments
        if "course_structure" in goal["topics"] and any("|" in ln for ln in sec["lines"]):
            f *= 1.5
        return f

    def _format_strict_sections(self, selected_secs, goal):
        asked_progs = goal["programs"]
        asked_topics = goal["topics"]
        output = []
        seen_lines = set()

        for sec in selected_secs:
            heading = sec["heading"].strip(" -—:")
            sec_lines = []

            for ln in sec["lines"]:
                clean = ln.strip()
                if not clean:
                    continue

                # Line-level course guard: never let another course's line slip in!
                if not self._valid_line_for_program(clean, asked_progs, asked_topics):
                    continue

                norm = re.sub(r'\W+', '', clean.replace("##SUBSECTION##", "").lower())
                if len(norm) < 2 or norm in seen_lines:
                    continue
                seen_lines.add(norm)

                if norm == re.sub(r'\W+', '', heading.lower()):
                    continue

                sec_lines.append(self._format_clean_line(clean))

            if sec_lines:
                if heading:
                    output.append(f"**{heading}**")
                output.append("\n".join(sec_lines))
                output.append("")

        return "\n".join(output).strip()

    def _format_faculty_profile(self, fac):
        lines = fac["lines"]
        header = fac["full_name"].lstrip("•-* ").strip()
        out = [f"**{header}**\n"]
        seen = {re.sub(r'\W+', '', header.lower())}

        for ln in lines[1:]:
            clean = ln.replace("##SUBSECTION##", "").strip()
            norm = re.sub(r'\W+', '', clean.lower())
            if len(norm) < 2 or norm in seen:
                continue
            seen.add(norm)
            out.append(self._format_clean_line(clean))

        return "\n".join(out).strip()

    def _format_clean_line(self, line):
        clean = line.strip()
        if clean.startswith("##SUBSECTION##"):
            sub_title = clean.replace("##SUBSECTION##", "").strip().rstrip(":")
            return f"\n**{sub_title}**"

        clean = clean.lstrip("•-* ").strip()
        if "|" in clean:
            parts = [p.strip() for p in clean.split("|") if p.strip()]
            return "• " + " — ".join(parts)
        if clean.endswith(":") and len(clean) < 95:
            return f"\n**{clean}**"
        if ":" in clean and len(clean) < 140:
            k, v = clean.split(":", 1)
            if len(k.split()) <= 6:
                return f"• **{k.strip()}:** {v.strip()}"
        return f"• {clean}"