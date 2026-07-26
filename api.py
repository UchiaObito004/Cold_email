"""
Self-contained FastAPI service: job-matching + AI cold-email generation.

Run with:
    uvicorn api:app --reload

Endpoints:
    POST /jobs/extract   - parse a job posting (URL or pasted text) into structured JSON
    POST /resume/parse   - upload a resume PDF, get back extracted projects + sender profile
    POST /match          - match resume projects against a job's skills
    POST /email/generate - generate the final cold email from a job + matched projects + sender info
    POST /apply          - full pipeline in one call: job + resume + sender details -> email
    GET  /health         - liveness check
"""

import os
import re
import uuid
import hashlib
import tempfile
from typing import Optional
from difflib import SequenceMatcher

import requests
import chromadb
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=2048,
    api_key=os.getenv("GROQ_API_KEY"),
)

chroma_client = chromadb.EphemeralClient()


def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'http\S+\s*', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def scrape_job_posting(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return clean_text(soup.get_text(separator=" "))


def get_job_text(input_type: str, value: str) -> str:
    if input_type == "url":
        return scrape_job_posting(value)
    elif input_type == "text":
        return clean_text(value)
    raise ValueError("input_type must be 'url' or 'text'")


extract_prompt = PromptTemplate.from_template(
    """
    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}

    ### INSTRUCTION:
    The text above is scraped from a company's careers page or pasted from a job posting.
    Extract the job posting(s) and return them in **valid JSON only** with these keys:
    'role', 'experience', 'skills', 'description'.

    For 'skills': read the ENTIRE posting, including Responsibilities,
    Qualifications, Requirements, Preferred Skills, Nice-to-have Skills,
    and any technical examples. Extract EVERY programming language,
    framework, library, API, database, cloud platform, AI/ML technology,
    LLM, tool, software, protocol, automation platform, and technical skill
    explicitly mentioned. Do not infer skills that are not present.

    Many postings end with a short auto-generated tag
    line like "Skills: python, hiring" - that line is INCOMPLETE and must NOT be used as the
    sole source; always cross-check it against the full body and include anything mentioned
    there too (e.g. if the body says "Flask", "REST APIs", "SQL", "Git", include all of those
    even if the trailing tag line only says "python").

    'skills' MUST be a JSON array of individual skill strings, e.g.
    ["Python", "Flask", "REST APIs", "SQL", "Git"] - never a single comma-separated string.

    Only return the JSON. No preamble, no explanation, no markdown fences.

    ### VALID JSON (NO PREAMBLE):
    """
)

json_parser = JsonOutputParser()


def normalize_skills(skills) -> list:
    """Defensively coerce 'skills' into a clean list of strings, no matter what shape
    the LLM returned it in (list, comma-separated string, single string, None, etc.).
    This prevents the classic bug of iterating over a string's individual characters."""
    if skills is None:
        return []
    if isinstance(skills, list):
        return [str(s).strip() for s in skills if str(s).strip()]
    if isinstance(skills, str):
        return [s.strip() for s in re.split(r'[,;/]', skills) if s.strip()]
    return [str(skills)]


def extract_job(page_text: str) -> dict:
    chain = extract_prompt | llm
    raw_response = chain.invoke({"page_data": page_text})
    try:
        parsed = json_parser.parse(raw_response.content)
    except OutputParserException:
        retry_prompt = PromptTemplate.from_template(
            "Fix this into ONE valid JSON object with keys "
            "'role','experience','skills','description'. 'skills' must be a JSON array "
            "of strings, not a comma-separated string. "
            "Return ONLY the JSON, nothing else:\n\n{bad_output}"
        )
        retry_chain = retry_prompt | llm
        fixed = retry_chain.invoke({"bad_output": raw_response.content})
        parsed = json_parser.parse(fixed.content)

    if isinstance(parsed, list):
        if not parsed:
            raise ValueError(
                "Couldn't find job details in that text. If you used a URL, try pasting "
                "the job description text instead - some sites load content via "
                "JavaScript that scraping can't see."
            )
        parsed = parsed[0]

    parsed["skills"] = normalize_skills(parsed.get("skills"))
    return parsed


def extract_resume_text(uploaded_pdf) -> str:
    """Extract raw text from an uploaded resume PDF (Streamlit UploadedFile)."""
    reader = PdfReader(uploaded_pdf)
    raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not raw.strip():
        raise ValueError(
            "Could not extract any text from this PDF - it may be a scanned image "
            "rather than selectable text."
        )
    return raw.strip()


# ---------------- Sender profile (fixes hardcoded "recent graduate") ----------------
# Works for ANY person - any degree, branch/major, college, or graduation year.
# Extracts structured facts first (so nothing gets assumed or defaulted), then builds
# the sentence from those facts rather than trusting the LLM to phrase it correctly
# for a resume format it hasn't seen before.

profile_prompt = PromptTemplate.from_template(
    """
    ### RESUME TEXT:
    {resume_text}

    ### INSTRUCTION:
    Read the resume above and extract the sender's CURRENT education/career status as
    structured JSON. This resume could belong to ANY person - any degree, any branch/major
    (e.g. Computer Science, Mechanical, Commerce, Design, Medicine, etc.), any college, and
    any graduation year. Do NOT assume a specific field of study - extract exactly what is
    written for THIS resume.

    Return ONLY this JSON object:
    {{
        "status": one of "student", "recent_graduate", "professional", or "unclear",
        "degree": the degree/qualification exactly as written (e.g. "B.Tech", "B.Com",
                  "MBBS", "M.Sc") or "" if not stated,
        "branch": the branch/major/specialization exactly as written (e.g. "Mechanical
                  Engineering", "Finance", "Computer Science") or "" if not stated,
        "college": institution name exactly as written, or "" if not stated,
        "graduation_year": the graduation year (expected or completed) exactly as written,
                            or "" if not stated,
        "current_title": current job title if the resume shows they're employed, or "" if not,
        "years_experience": years of professional experience if stated, or "" if not
    }}

    Set "status":
    - "student" if there's a future/expected graduation year or the resume otherwise
      indicates they haven't graduated yet.
    - "recent_graduate" only if the resume clearly shows they already graduated and have
      little to no professional experience.
    - "professional" if the resume shows a current job title and/or years of experience.
    - "unclear" if none of the above can be determined confidently.

    Only return the JSON object. No preamble, no explanation, no markdown fences.
    """
)


def extract_sender_profile(resume_text: str) -> dict:
    chain = profile_prompt | llm
    raw_response = chain.invoke({"resume_text": resume_text})
    try:
        parsed = json_parser.parse(raw_response.content)
    except OutputParserException:
        retry_prompt = PromptTemplate.from_template(
            "Fix this into ONE valid JSON object with keys 'status','degree','branch',"
            "'college','graduation_year','current_title','years_experience'. "
            "Return ONLY the JSON, nothing else:\n\n{bad_output}"
        )
        retry_chain = retry_prompt | llm
        fixed = retry_chain.invoke({"bad_output": raw_response.content})
        parsed = json_parser.parse(fixed.content)
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    return parsed if isinstance(parsed, dict) else {}


def build_background_sentence(profile: dict) -> str:
    """Builds the sentence in plain Python from extracted facts - deterministic, and
    works correctly for any branch/college/year without relying on LLM phrasing."""
    status = profile.get("status", "unclear")
    degree = profile.get("degree", "")
    branch = profile.get("branch", "")
    college = profile.get("college", "")
    grad_year = profile.get("graduation_year", "")
    title = profile.get("current_title", "")
    years_exp = profile.get("years_experience", "")

    degree_branch = " ".join(x for x in [degree, f"in {branch}" if branch else ""] if x).strip()

    if status == "professional" and title:
        exp_part = f" with {years_exp} of experience" if years_exp else ""
        return f"{title}{exp_part}"
    if status == "student" and degree_branch:
        year_part = f", graduating in {grad_year}" if grad_year else ""
        return f"a student pursuing {degree_branch}{year_part}"
    if status == "recent_graduate" and degree_branch:
        return f"a recent {degree_branch} graduate"
    if degree_branch:
        return f"a candidate with a background in {degree_branch}"
    return "a candidate with relevant technical experience"


# ---------------- Resume project extraction ----------------

resume_project_prompt = PromptTemplate.from_template(
    """
    ### RESUME TEXT:
    {resume_text}

    ### INSTRUCTION:

    Extract every project mentioned in this resume.

    For each project return a JSON object with:

    - "name":
      Project name exactly as written (or as close as possible).

    - "techstack":
      A single space-separated string containing ONLY the technologies,
      programming languages, frameworks, libraries, databases, cloud
      platforms, APIs, and tools that are EXPLICITLY mentioned for that
      project. Do NOT invent technologies that are not written in the resume.

    - "description":
      A 1-2 sentence summary of what the project actually does and its key
      outcome/result, based ONLY on the resume's bullet points for that project
      (e.g. what it predicts, what it automates, what accuracy/metric it achieved).
      Do NOT invent achievements, metrics, or functionality not stated in the resume.

    - "link":
      GitHub / Portfolio / Live URL if present. Otherwise return "".

    Return ONLY a valid JSON array.

    Example:

    [
        {{
            "name": "Loan Default Predictor",
            "techstack": "Python Scikit-learn FastAPI Docker MLflow Optuna",
            "description": "An end-to-end MLOps pipeline that predicts loan default risk, comparing 6 ML algorithms and serving predictions via a FastAPI REST endpoint with 98.41% accuracy.",
            "link": "https://github.com/..."
        }}
    ]

    No markdown. No explanation. No extra text.
    """
)


def extract_projects_from_resume(resume_text: str) -> list:
    chain = resume_project_prompt | llm
    raw_response = chain.invoke({"resume_text": resume_text})
    try:
        parsed = json_parser.parse(raw_response.content)
    except OutputParserException:
        retry_prompt = PromptTemplate.from_template(
            "Fix this into ONE valid JSON array of objects with keys "
            "'name','techstack','description','link'. Return ONLY the JSON array, "
            "nothing else:\n\n{bad_output}"
        )
        retry_chain = retry_prompt | llm
        fixed = retry_chain.invoke({"bad_output": raw_response.content})
        parsed = json_parser.parse(fixed.content)
    if isinstance(parsed, dict):
        parsed = [parsed]
    return [p for p in parsed if isinstance(p, dict) and p.get("techstack")]


def build_portfolio_collection(projects: list, session_id: str):
    collection_name = f"portfolio_{session_id}"

    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    session_collection = chroma_client.create_collection(name=collection_name)

    for p in projects:
        name = p.get("name", "Project")
        techstack = p.get("techstack", "")
        description = p.get("description", "")
        link = p.get("link", "") or ""

        # Richer embedding text so semantic search also picks up on what the
        # project does, not just its raw tech keywords.
        document = f"""Project Name: {name}
Technologies: {techstack}
Summary: {description}"""

        row_id = hashlib.md5((name + techstack).encode()).hexdigest()

        session_collection.add(
            documents=[document],
            metadatas=[{
                "name": name,
                "techstack": techstack,
                "description": description,
                "links": link,
            }],
            ids=[row_id],
        )

    return session_collection


def query_session_portfolio(session_collection, skills: list, n_results: int = 10) -> list:
    """Vector-search the portfolio for candidate projects.

    Returns:
        [{"name", "techstack", "description", "link", "similarity"}, ...]
    """
    query_text = ", ".join(skills) if isinstance(skills, list) else str(skills)

    total_projects = session_collection.count()
    if total_projects == 0:
        return []

    n = min(n_results, total_projects)

    results = session_collection.query(
        query_texts=[query_text],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    projects = []
    for meta, distance in zip(metas, distances):
        similarity = 1 / (1 + float(distance))
        projects.append({
            "name": meta.get("name", "Project"),
            "techstack": meta.get("techstack", ""),
            "description": meta.get("description", ""),
            "link": meta.get("links", ""),
            "similarity": similarity,
        })

    return projects


SKILL_MAP = {
    "langchain": [
        "llm", "large language models", "prompt engineering", "generative ai",
        "ai", "genai", "ai application development", "rag frameworks",
        "retrieval augmented generation",
    ],
    "llama-3": ["llm", "large language models", "generative ai", "genai"],
    "groq": ["llm", "large language models", "generative ai", "genai"],
    "rag": [
        "retrieval augmented generation", "retrieval", "generative ai", "llm",
        "semantic search", "vectorization", "document processing",
        "chunking strategies", "context augmentation",
        "grounded response generation",
    ],
    "chromadb": [
        "vector database", "retrieval", "rag", "vectorization",
        "semantic search", "embeddings",
    ],
    "pydantic": ["structured outputs", "data validation"],
    "fastapi": ["api", "apis", "rest api"],
    "scikit-learn": ["machine learning", "ml"],
    "tensorflow": ["deep learning", "machine learning"],
    "streamlit": ["web app", "dashboard"],
    "docker": ["containerization"],
    "mlflow": ["experiment tracking", "mlops"],
    "optuna": ["hyperparameter tuning"],
    "github actions": ["ci", "cd", "automation"],
    "dvc": ["data versioning", "mlops"],
}


def _expand_tokens(tokens: set) -> set:
    """Expand a set of single-word tokens using SKILL_MAP synonyms, tokenizing each
    synonym phrase into individual words. This matters because a synonym like "large
    language models" needs to be split into {"large","language","models"} to actually
    overlap with a job skill phrased the same way - adding it as one un-split phrase
    (as the old per-function `expand()` helpers did) silently never matches anything."""
    expanded = set(tokens)
    for token in list(tokens):
        if token in SKILL_MAP:
            for synonym in SKILL_MAP[token]:
                expanded.update(
                    w.lower() for w in re.split(r'[\s,/()+\-]+', synonym) if len(w) > 1
                )
    return expanded


def filter_relevant_projects(candidates: list, job_skills: list, min_score: float = 6.0) -> list:
    """Combines ChromaDB semantic similarity + exact skill overlap + synonym expansion
    into one relevance score, so the strongest, most defensible projects surface first."""

    def tokenize(text):
        return {w.lower() for w in re.split(r'[\s,/()+\-]+', text) if len(w) > 1}

    def fuzzy_overlap(a, b):
        overlap = 0
        used = set()
        for x in a:
            if x in b:
                overlap += 1
                used.add(x)
                continue
            for y in b - used:
                if SequenceMatcher(None, x, y).ratio() >= 0.85:
                    overlap += 1
                    used.add(y)
                    break
        return overlap

    job_tokens = _expand_tokens(tokenize(", ".join(normalize_skills(job_skills))))

    ranked = []
    for project in candidates:
        tech_tokens = _expand_tokens(tokenize(project["techstack"]))
        overlap = fuzzy_overlap(job_tokens, tech_tokens)
        similarity = project.get("similarity", 0)
        final_score = similarity * 10 + overlap

        project["score"] = round(final_score, 2)
        project["overlap"] = overlap

        if final_score >= min_score:
            ranked.append(project)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


# ---------------- Coverage gap detection ----------------
# Catches the case where a JD spans multiple skill categories (e.g. frontend + backend)
# but the matched projects only genuinely cover some of them. Without this, the email
# writer has no way to know it's overselling relevance for categories it doesn't cover.

CATEGORY_KEYWORDS = {
    "Frontend": [
        "react", "angular", "vue", "typescript", "javascript", "html", "css",
        "material-ui", "materialui", "gatsby", "nextjs", "redux", "tailwind",
        "frontend", "ui", "ux", "svelte", "webpack",
    ],
    "Backend": [
        "flask", "django", "spring", "express", "backend", "microservices",
        "nodejs", "node", "fastapi", "api", "apis", "rest", "restful", "graphql",
    ],
    "Database": [
        "sql", "mysql", "postgres", "postgresql", "mongodb", "dynamodb", "nosql",
        "database", "redis", "cassandra", "sqlite",
    ],
    "DevOps/Cloud": [
        "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "jenkins",
        "devops", "cloud", "ci", "cd",
    ],
    "ML/AI": [
        "machine", "learning", "deep", "tensorflow", "pytorch", "nlp", "llm",
        "llms", "langchain", "rag", "ai", "scikit-learn", "genai", "keras",
        "generative", "prompt", "embeddings", "embedding", "vectorization",
        "vector", "chromadb", "langgraph", "llamaindex", "autogen", "crewai",
        "semantic", "agentic", "chatbot", "transformer", "transformers",
        "gpt", "llama", "mistral", "tokenization", "huggingface",
    ],
    "Mobile": [
        "ios", "android", "swift", "kotlin", "flutter", "mobile",
    ],
    "Security": [
        "security", "cybersecurity", "pentest", "penetration", "vulnerability",
        "siem", "soc", "firewall", "encryption", "cryptography", "malware",
        "forensics", "nmap", "metasploit", "wireshark", "burp", "splunk",
        "nessus", "qualys", "ids", "ips", "cissp", "ceh", "comptia", "iam",
        "threat", "phishing", "zero-trust", "sso",
    ],
}


def _categorize_text(text: str) -> set:
    tokens = {w.lower() for w in re.split(r'[\s,/()+\-]+', text) if len(w) > 1}
    hit_categories = set()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if tokens & set(keywords):
            hit_categories.add(category)
    return hit_categories


def _skill_present(skill_tokens: set, tech_tokens: set) -> bool:
    expanded_tech_tokens = _expand_tokens(tech_tokens)
    if skill_tokens & expanded_tech_tokens:
        return True
    for s in skill_tokens:
        for t in expanded_tech_tokens:
            if SequenceMatcher(None, s, t).ratio() >= 0.85:
                return True
    return False


def find_coverage_gaps(job_skills: list, matched_links: list, min_coverage_ratio: float = 0.4) -> dict:
    """Returns {category: [uncovered job skills]} for every skill category where the
    matched projects cover FEWER than min_coverage_ratio of that category's specific
    JD skills. This is deliberately per-skill, not per-category-keyword: previously,
    a project sharing just ONE tool (e.g. Docker) with a JD would mark the ENTIRE
    "DevOps/Cloud" category as "covered" even if Kubernetes, Terraform, AWS, Ansible,
    and monitoring were all still missing. Now each category needs real proportional
    overlap, not a single coincidental shared tool."""
    job_skills = normalize_skills(job_skills)

    job_categories = {}
    for skill in job_skills:
        skill_tokens = {w.lower() for w in re.split(r'[\s,/()+\-]+', skill) if len(w) > 1}
        for category, keywords in CATEGORY_KEYWORDS.items():
            if skill_tokens & set(keywords):
                job_categories.setdefault(category, []).append(skill)

    tech_tokens = set()
    for p in matched_links:
        tech_tokens |= {w.lower() for w in re.split(r'[\s,/()+\-]+', p.get("techstack", "")) if len(w) > 1}

    gaps = {}
    for category, skills in job_categories.items():
        covered = [
            s for s in skills
            if _skill_present({w.lower() for w in re.split(r'[\s,/()+\-]+', s) if len(w) > 1}, tech_tokens)
        ]
        uncovered = [s for s in skills if s not in covered]
        coverage_ratio = len(covered) / len(skills) if skills else 1.0
        if coverage_ratio < min_coverage_ratio:
            gaps[category] = uncovered

    return gaps


def overall_skill_coverage(job_skills: list, matched_links: list) -> float:
    """Fraction of the JD's specific skills that are genuinely present (exact or fuzzy)
    across the matched projects combined. Used as a soft signal alongside
    best_category_coverage() below: a broad, multi-domain JD (e.g. AI/ML + backend +
    frontend + cloud + database) will naturally have low OVERALL coverage even from a
    great domain-specialist candidate, so this alone should never zero out an
    otherwise strong, focused match."""
    job_skills = normalize_skills(job_skills)
    if not job_skills:
        return 1.0

    tech_tokens = set()
    for p in matched_links:
        tech_tokens |= {w.lower() for w in re.split(r'[\s,/()+\-]+', p.get("techstack", "")) if len(w) > 1}

    matched = 0
    for skill in job_skills:
        skill_tokens = {w.lower() for w in re.split(r'[\s,/()+\-]+', skill) if len(w) > 1}
        if _skill_present(skill_tokens, tech_tokens):
            matched += 1

    return matched / len(job_skills)


def best_category_coverage(
    job_skills: list,
    matched_links: list,
    min_category_share: float = 0.2,
    min_category_skills: int = 3,
):
    """Returns (best_category_name, ratio) - the single JD skill category the matched
    projects cover most thoroughly. A job posting that spans several domains (e.g. this
    QuantumLoopAI listing needs AI/ML *and* NestJS backend *and* Next.js frontend *and*
    Azure/MySQL) will always drag overall_skill_coverage() down even for a candidate who
    is a near-perfect match for the role's actual core skill (AI/ML). Gating purely on
    overall coverage therefore produces false "no match" results for genuine
    domain-specialist candidates. This function lets the app instead ask: "is there at
    least one real skill area this candidate's projects strongly demonstrate?"

    ELIGIBILITY GUARD: a category only qualifies as a possible "anchor" if it makes up
    at least min_category_share of the JD's total listed skills AND has at least
    min_category_skills skills in absolute terms. Without this, a JD that is
    fundamentally about one domain (e.g. a cybersecurity role that just happens to
    mention Docker and Kubernetes in passing) could be "matched" purely because that
    tiny, incidental 2-skill DevOps/Cloud slice is fully covered - even though the
    role's actual core skills (pentesting, SIEM, vulnerability assessment, etc.) aren't
    covered at all. A category has to represent a real, substantial theme of the
    posting - not a couple of stray keywords - before it's allowed to validate a match.
    """
    job_skills = normalize_skills(job_skills)
    total_skills = len(job_skills)
    if total_skills == 0:
        return None, 0.0

    job_categories = {}
    for skill in job_skills:
        skill_tokens = {w.lower() for w in re.split(r'[\s,/()+\-]+', skill) if len(w) > 1}
        for category, keywords in CATEGORY_KEYWORDS.items():
            if skill_tokens & set(keywords):
                job_categories.setdefault(category, []).append(skill)

    tech_tokens = set()
    for p in matched_links:
        tech_tokens |= {w.lower() for w in re.split(r'[\s,/()+\-]+', p.get("techstack", "")) if len(w) > 1}

    best_category, best_ratio = None, 0.0
    for category, skills in job_categories.items():
        category_share = len(skills) / total_skills
        if category_share < min_category_share or len(skills) < min_category_skills:
            continue  # too small/incidental a slice of the JD to anchor a match on

        covered = [
            s for s in skills
            if _skill_present({w.lower() for w in re.split(r'[\s,/()+\-]+', s) if len(w) > 1}, tech_tokens)
        ]
        ratio = len(covered) / len(skills)
        if ratio > best_ratio:
            best_category, best_ratio = category, ratio

    return best_category, best_ratio


def format_coverage_gaps(gaps: dict) -> str:
    if not gaps:
        return "(none - the matched projects reasonably cover this JD's key skill areas)"
    lines = [
        f"- {category}: {', '.join(skills)} (NOT covered by any matched project)"
        for category, skills in gaps.items()
    ]
    return "\n".join(lines)


# ---------------- Email generation ----------------

email_prompt = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    {job_description}

    ### SENDER DETAILS:
    Name: {sender_name}
    Phone: {sender_phone}
    Email: {sender_email}
    LinkedIn: {sender_linkedin}
    Background: {sender_background}

    ### RELEVANT PROJECTS (the ONLY projects and skills you may mention - each line is
    "Name: <project name> | Techstack: <skills used> | Summary: <what it does> | GitHub: <link, if any>"):
    {link_list}

    ### SKILL AREAS THIS JOB NEEDS THAT YOUR PROJECTS DO NOT COVER:
    {coverage_gaps}

    ### INSTRUCTION:
    Write a formal, well-structured cold email applying for the role above, following this
    EXACT format:

    1. Subject line: "Subject: Application for <Role> - <one-line hook>"
    2. Blank line, then a greeting. Use "Dear <Company> Team," ONLY if the company name is
       LITERALLY written somewhere in the JOB DESCRIPTION text above. Do NOT guess, infer,
       or recall a company name from outside knowledge (e.g. a firm you recognize from
       training data) - if the company name does not appear verbatim in the job description
       text, you MUST use "Dear Hiring Manager," instead. Never invent a company name.
    3. Opening paragraph: 2-3 sentences introducing the sender. Explicitly work in the exact
       "Background" field above (e.g. "As <Background>, I..." or "I am <Name>, <Background>,
       and...") to state their real education/experience status - do not write something
       vague instead, and do not default to "recent graduate" unless Background says so.
       Do NOT invent or substitute a different role/title for the sender (e.g. "a developer",
       "a backend engineer", "a software engineer") - use the Background field's wording, not
       a generalization of it. Only mention skills that appear in the RELEVANT PROJECTS
       techstack list above - never mention a tool, framework, or skill that isn't literally
       listed there.
    4. For EACH project listed in RELEVANT PROJECTS above (and ONLY those - do not add,
       skip, or invent any project), a separate block in this exact shape:
       - The project's "Name" from the list, bolded, on its own line (use **Name** markdown-style bold)
       - 1-2 sentences that: (a) explain what the project actually does, using its "Summary"
         field as the basis, (b) name which technologies from its "Techstack" field were used,
         and (c) connect it to a specific job requirement where relevant, using concepts like
         workflow automation, LLM integration, prompt engineering, retrieval, APIs, data
         processing, or analytics - ONLY where the project and job description genuinely
         support that connection. Do NOT invent technologies or achievements not present in
         the Summary or Techstack fields. Do NOT use architecture/competency-level claims
         (e.g. "microservices", "backend development expertise", "scalable backend
         solutions", "production-grade backend systems") unless that exact word or a close
         variant is literally present in the project's Techstack or Summary field - describe
         precisely what was built (e.g. "served predictions through a FastAPI endpoint") not
         a broader professional specialty the project doesn't actually demonstrate.
       - CRITICAL: never use strong-relevance language ("directly aligns with", "key
         requirement for", "perfectly suited for", "core requirement") to describe a
         project's fit with any skill area listed in "SKILL AREAS THIS JOB NEEDS THAT YOUR
         PROJECTS DO NOT COVER" above. Only use that kind of language for skills genuinely
         present in the project's own Techstack field.
       - If that project has a GitHub link, put it on its OWN line as "GitHub: <url>".
         If it has no link, omit this line entirely for that project - do NOT write
         "GitHub: N/A" or similar placeholder text.
       - A blank line before the next project block
    5. A closing paragraph (2-3 sentences) expressing enthusiasm and availability. Do NOT
       name, list, or reference any specific skill/technology/tool that isn't literally in
       the RELEVANT PROJECTS list above - not even in an "eager to learn X" framing. Never
       volunteer what the sender lacks; only present what they actually have.
    6. "Thank you for considering my application." on its own line, followed by an offer to
       discuss further.
    7. Sign-off "Best regards," on its own line, then a blank line, then the sender's contact
       block with EACH item on its own separate line, in this order:
       Name
       Phone
       Email
       LinkedIn: <url>

    Do not merge the contact details onto one line. Do not add any preamble before the
    "Subject:" line. Do not invent facts, skills, or projects not present in the sender
    details, job description, or the RELEVANT PROJECTS list above.

    ### EMAIL (NO PREAMBLE):
    """
)


def _format_links(links: list) -> str:
    if not links:
        return "(none provided - do not include a project block)"
    lines = []
    for p in links:
        line = f"Name: {p.get('name', 'Project')} | Techstack: {p.get('techstack', '')}"
        if p.get("description"):
            line += f" | Summary: {p['description']}"
        if p.get("link"):
            line += f" | GitHub: {p['link']}"
        lines.append(line)
    return "\n".join(lines)


def generate_email(job: dict, links: list, sender_info: dict, coverage_gaps: dict = None) -> str:
    chain = email_prompt | llm
    response = chain.invoke({
        "job_description": str(job),
        "link_list": _format_links(links),
        "coverage_gaps": format_coverage_gaps(coverage_gaps or {}),
        "sender_name": sender_info.get("name", ""),
        "sender_phone": sender_info.get("phone", ""),
        "sender_email": sender_info.get("email", ""),
        "sender_linkedin": sender_info.get("linkedin", ""),
        "sender_background": sender_info.get("background", "a candidate with relevant technical experience"),
    })
    return response.content


critique_prompt = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    {job_description}

    ### RELEVANT PROJECTS (the ONLY facts/skills about the sender that may be used):
    {link_list}

    ### SKILL AREAS THIS JOB NEEDS THAT THE PROJECTS DO NOT COVER:
    {coverage_gaps}

    ### DRAFT EMAIL:
    {draft_email}

    ### INSTRUCTION:
    Check the draft email above ONLY for hallucination AND overstated relevance:
    - Any skill, tool, or framework mentioned that is NOT literally listed in a project's
      "Techstack:" field above.
    - Any project mentioned that is NOT in the RELEVANT PROJECTS list above.
    - Invented metrics, invented experience, or invented company names - including in the
      greeting line (e.g. "Dear X Team,"). If the company name in the greeting does NOT
      appear verbatim anywhere in the JOB DESCRIPTION above, that is a hallucination and
      must be flagged; the greeting should be "Dear Hiring Manager," instead.
    - Any claim about the sender's education/career status that contradicts or overstates
      what a reasonable reading of the job/sender context would support.
    - OVERSTATED RELEVANCE: strong-fit language ("directly aligns with", "key requirement
      for", "perfectly suited for", "core requirement") applied to any skill area listed in
      "SKILL AREAS THIS JOB NEEDS THAT THE PROJECTS DO NOT COVER" above. This is misleading
      even though no fact is technically invented, and must be flagged.
    - INVENTED ROLE/TITLE: the opening paragraph describing the sender as "a developer",
      "an engineer", or any professional title/role not literally present in how their
      background was described - flag this even if every individual skill mentioned is real.
    - ARCHITECTURE OVERCLAIM: words like "microservices", "backend development expertise",
      "scalable backend solutions", or similar competency-level claims that are NOT literally
      present in any matched project's Techstack/Summary field - using a single API endpoint
      to serve a model is not the same as claiming general backend/microservices expertise.
    - NAMED MISSING SKILLS: the email naming, listing, or referencing ANY specific
      skill/technology/tool from "SKILL AREAS THIS JOB NEEDS THAT THE PROJECTS DO NOT COVER"
      above, even in an "eager to learn X" or "looking to grow into Y" framing. The sender
      should never volunteer what they lack - flag any such mention for removal.
    Do NOT comment on tone or style.
    If it's fully grounded and honest about gaps, respond with exactly: OK
    Otherwise, list each issue on its own line, prefixed with "- ".

    ### RESULT:
    """
)

refine_prompt = PromptTemplate.from_template(
    """
    ### ORIGINAL DRAFT:
    {draft_email}

    ### SENDER'S ACTUAL BACKGROUND (must appear accurately in the opening paragraph -
    do not substitute a different role/title like "a developer" or "an engineer"):
    {sender_background}

    ### ISSUES FOUND (fix these, remove anything not grounded in the portfolio/job description):
    {critique}

    ### INSTRUCTION:
    Rewrite the email, removing or correcting every issue listed above.
    Keep everything else about the draft the same - same length, same tone.
    IMPORTANT: preserve the exact structure of the original draft - subject line, bolded
    project name on its own line per project, each project's link on its own "GitHub: <url>"
    line with a blank line between projects, and the closing contact block with Name, Phone,
    Email, and LinkedIn each on their own separate line. The opening paragraph MUST describe
    the sender using the SENDER'S ACTUAL BACKGROUND text above, word-for-word or a close
    paraphrase of it - never a generic substitute role.
    Do not add a preamble.

    ### FINAL EMAIL (NO PREAMBLE):
    """
)


def generate_checked_email(job: dict, links: list, sender_info: dict) -> str:
    coverage_gaps = find_coverage_gaps(job.get("skills", []), links)

    draft = generate_email(job, links, sender_info, coverage_gaps)

    formatted_links = _format_links(links)

    critique_chain = critique_prompt | llm
    critique = critique_chain.invoke({
        "job_description": str(job),
        "link_list": formatted_links,
        "coverage_gaps": format_coverage_gaps(coverage_gaps),
        "draft_email": draft,
    }).content.strip()

    if critique.upper() == "OK":
        return draft

    refine_chain = refine_prompt | llm
    final_email = refine_chain.invoke({
        "draft_email": draft,
        "critique": critique,
        "sender_background": sender_info.get("background", "a candidate with relevant technical experience"),
    }).content
    return final_email
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Cold Email Generator API",
    description="Job-matching + AI cold-email generation, as a service.",
    version="1.0.0",
)

# Adjust origins as needed for your frontend before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MIN_OVERALL_COVERAGE = 0.25
MIN_BEST_CATEGORY_COVERAGE = 0.5


# ---------------- Schemas ----------------

class JobExtractRequest(BaseModel):
    input_type: str = Field(..., description="'url' or 'text'")
    value: str = Field(..., description="Job posting URL, or pasted job description text")


class SenderInfo(BaseModel):
    name: str
    phone: str = ""
    email: str = ""
    linkedin: str = ""
    background: str = "a candidate with relevant technical experience"


class MatchRequest(BaseModel):
    projects: list = Field(..., description="Projects as returned by /resume/parse")
    job_skills: list = Field(..., description="Skills as returned by /jobs/extract")
    session_id: Optional[str] = Field(
        None, description="Reuse a previous session_id to avoid rebuilding the portfolio index"
    )


class EmailGenerateRequest(BaseModel):
    job: dict
    links: list = Field(default_factory=list, description="Matched projects to reference in the email")
    sender_info: SenderInfo


# ---------------- Helpers ----------------

def _compute_match(job_skills: list, projects: list, session_id: str):
    collection = build_portfolio_collection(projects, session_id)
    candidates = query_session_portfolio(collection, job_skills)
    links = filter_relevant_projects(candidates, job_skills)
    links = links[:2]  # top 2 best-matching projects

    coverage_ratio = overall_skill_coverage(job_skills, links)
    best_category, best_category_ratio = best_category_coverage(job_skills, links)

    matched = bool(links) and not (
        coverage_ratio < MIN_OVERALL_COVERAGE and best_category_ratio < MIN_BEST_CATEGORY_COVERAGE
    )
    if not matched:
        links = []

    gaps = find_coverage_gaps(job_skills, links)

    return {
        "candidates": candidates,
        "links": links,
        "matched": matched,
        "coverage_ratio": coverage_ratio,
        "best_category": best_category,
        "best_category_ratio": best_category_ratio,
        "coverage_gaps": gaps,
    }


# ---------------- Endpoints ----------------

@app.post("/jobs/extract")
def extract_job_endpoint(payload: JobExtractRequest):
    """Parse a job posting (by URL or pasted text) into {role, experience, skills, description}."""
    try:
        page_text = get_job_text(payload.input_type, payload.value)
        job = extract_job(page_text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to read/parse job posting: {e}")
    return job


@app.post("/resume/parse")
async def parse_resume_endpoint(file: UploadFile = File(...)):
    """Upload a resume PDF. Returns extracted projects, sender profile, and a ready-to-use
    background sentence for the email's opening paragraph."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            resume_text = extract_resume_text(f)
        projects = extract_projects_from_resume(resume_text)
        sender_profile = extract_sender_profile(resume_text)
        sender_background = build_background_sentence(sender_profile)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return {
        "resume_text": resume_text,
        "projects": projects,
        "sender_profile": sender_profile,
        "sender_background": sender_background,
    }


@app.post("/match")
def match_endpoint(payload: MatchRequest):
    """Match a resume's projects against a job's skills. Returns the top matched projects
    plus coverage stats explaining why the match did or didn't pass."""
    session_id = payload.session_id or str(uuid.uuid4())[:8]
    result = _compute_match(payload.job_skills, payload.projects, session_id)
    result["session_id"] = session_id
    return result


@app.post("/email/generate")
def generate_email_endpoint(payload: EmailGenerateRequest):
    """Generate the final, hallucination-checked cold email from a job, a set of matched
    projects (can be empty to generate without project evidence), and sender details."""
    try:
        email = generate_checked_email(payload.job, payload.links, payload.sender_info.dict())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to generate email: {e}")
    return {"email": email}


@app.post("/apply")
async def apply_endpoint(
    job_input_type: str = Form(..., description="'url' or 'text'"),
    job_value: str = Form(..., description="Job posting URL, or pasted job description text"),
    name: str = Form(...),
    phone: str = Form(""),
    email: str = Form(""),
    linkedin: str = Form(""),
    force: bool = Form(False, description="If true, generate the email even if no projects matched strongly"),
    resume: UploadFile = File(...),
):
    """Full pipeline in a single call: reads the job posting, parses the resume, matches
    projects, and returns the generated email - mirroring the Streamlit app's flow end to end.

    If no projects match strongly and `force` is not set, returns matched=false with the
    coverage details instead of an email, so the caller can decide whether to retry with
    force=true (equivalent to the Streamlit app's "Generate anyway" button).
    """
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF resume.")

    try:
        page_text = get_job_text(job_input_type, job_value)
        job = extract_job(page_text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to read/parse job posting: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            resume_text = extract_resume_text(f)
        projects = extract_projects_from_resume(resume_text)
        sender_profile = extract_sender_profile(resume_text)
        sender_background = build_background_sentence(sender_profile)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)

    sender_info = {
        "name": name,
        "phone": phone,
        "email": email,
        "linkedin": linkedin,
        "background": sender_background,
    }

    session_id = str(uuid.uuid4())[:8]
    match_result = _compute_match(job.get("skills", []), projects, session_id)

    if not match_result["matched"] and not force:
        return {
            "matched": False,
            "job": job,
            "projects": projects,
            "sender_background": sender_background,
            "coverage_ratio": match_result["coverage_ratio"],
            "best_category": match_result["best_category"],
            "best_category_ratio": match_result["best_category_ratio"],
            "message": (
                "None of the resume projects matched the job strongly enough. "
                "Retry with force=true to generate an email without project evidence."
            ),
        }

    links = match_result["links"]
    try:
        generated_email = generate_checked_email(job, links, sender_info)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to generate email: {e}")

    return {
        "matched": match_result["matched"],
        "job": job,
        "links": links,
        "coverage_gaps": match_result["coverage_gaps"],
        "sender_background": sender_background,
        "email": generated_email,
    }


@app.get("/health")
def health():
    return {"status": "ok"}