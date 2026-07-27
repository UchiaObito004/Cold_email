# 📧 AI Cold Email Generator

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-000000?style=flat&logo=langchain)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange?style=flat)
![GPT](https://img.shields.io/badge/GPT--OSS-120B-blueviolet?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

An AI tool that reads a job posting, looks at your resume, and writes you a cold email —
but only if your projects actually match the job. If they don't, it tells you honestly
instead of writing something misleading.


## 🖥️ Demo

![img1](https://github.com/UchiaObito004/Cold_email/blob/main/img1.png?raw=true)


![img2](https://github.com/UchiaObito004/Cold_email/blob/main/img2.png?raw=true)

**Live demo:** https://cold-email-7r0i.onrender.com

**Interactive docs:** https://cold-email-7r0i.onrender.com/docs

---

## 🧠 How it works

```
User provides Job URL/Text
          │
          ▼
BeautifulSoup extracts the job description (if URL)
          │
          ▼
LLM extracts structured information
(Job title, required skills, responsibilities)
          │
          ▼
Resume is parsed (pypdf)
          │
          ▼
Projects are converted into embeddings
and stored in ChromaDB
          │
          ▼
LangChain orchestrates the pipeline —
chaining the extraction prompts, the
ChromaDB retriever, and the validation
pass into a single flow
          │
          ▼
ChromaDB retrieves top-k relevant projects
via cosine similarity
          │
          ▼
Python calculates:
- Semantic similarity (from retrieval)
- Skill overlap (keyword-level rerank)
          │
          ▼
Final Match Score
          │
   Is score above threshold? (0.65 default)
         /                 \
      No                    Yes
      │                      │
Return:                  Generate
"Not a good             Cold Email
 match"                    │
                            ▼
                  Validation Check
                  (Second LLM pass)
                            │
                            ▼
          Is every claim supported by
               the resume/projects?
                   /             \
                 No               Yes
                 │                 │
        Remove/fix false       Return final
           statements            email
```

## ✨ Features

- 🔗 **Works with a job link or pasted text** — paste any job URL, or copy-paste the description directly
- 📄 **Reads your resume automatically** — upload a PDF, it pulls out your projects and background
- 🎯 **Honest matching** — refuses to generate an email if your projects don't genuinely fit the job
- 🤖 **Self-checking emails** — every email is checked by a second AI pass for made-up claims before it's returned
- 🐳 **Runs anywhere** — packaged as a Docker image, works on any machine or server
- ☁️ **Live and hosted** — already deployed and usable from any device, no install needed

---

## 🛠️ Tech stack

| Part | Technology |
|---|---|
| API framework | FastAPI |
| LLM | GPT--OSS-120B via Groq |
| Orchestration | LangChain |
| Vector search | ChromaDB |
| Resume reading | pypdf |
| Web scraping | BeautifulSoup + Requests |
| Packaging | Docker |
| Hosting | Render |

---

## 📁 Project structure

```
Cold_email/
├── api.py              # Everything — parsing, matching, email generation, API endpoints
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker build instructions
└── .env                 # Your Groq API key (not committed)
```

Everything lives in one file (`api.py`) on purpose — no extra files needed to run it.

---

## ⚙️ Setup & installation

### 1. Clone the repository
```bash
git clone https://github.com/UchiaObito004/Cold_email.git
cd Cold_email
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at 👉 [console.groq.com](https://console.groq.com)

### 4. Run it
```bash
uvicorn api:app --reload
```
Open `http://localhost:8000/docs` in your browser to try it out.

---

## 🐳 Run with Docker instead

No Python setup needed — just Docker:
```bash
docker pull ashborn004/cold-email-api:latest
docker run -p 8000:8000 -e GROQ_API_KEY=your_groq_api_key_here ashborn004/cold-email-api:latest
```
Then open `http://localhost:8000/docs`.

---

## 🚀 Usage

The easiest way to try it is through the interactive docs at `/docs` — it lets you fill in
a form and test any endpoint from your browser, no coding needed.

**Main endpoints:**

| Endpoint | What it does |
|---|---|
| `POST /jobs/extract` | Reads a job posting and pulls out the role, skills, and requirements |
| `POST /resume/parse` | Reads your resume PDF and pulls out your projects and background |
| `POST /match` | Checks how well your projects match a job's requirements |
| `POST /email/generate` | Writes the email from a job and your matched projects |
| `POST /apply` | Does all of the above in one step — job + resume in, email out |

`POST /apply` is the one most people want — give it a job (link or text), your resume PDF,
and your contact details, and it returns a ready-to-send email (or an honest "this isn't a
strong match" if your projects don't fit).

---

## 🔑 Key decisions, explained simply

**Why does it sometimes refuse to write an email?**
Because a cold email claiming skills you don't have doesn't help you — it just wastes the
recruiter's time and yours. The app checks real skill overlap before writing anything, and
tells you honestly if there isn't enough of a match.

**Why does every email get double-checked?**
AI can sometimes make things sound more impressive than they really are. A second pass
reviews the draft and removes anything that isn't actually true, before you ever see it.

**Why Groq?**
It's one of the fastest ways to run gpt-oss-120b — emails generate in seconds, not minutes.

**Why is everything in one file?**
So anyone can download `api.py`, drop it next to a `requirements.txt`, and run it — no
tracking down multiple files or figuring out imports.

---

## 🤝 Contributing

Pull requests are welcome. For bigger changes, open an issue first so we can talk it
through.

---

## 👤 Author

**Bhushan Verma**

- 🎓 B.Tech AI & Data Science — graduating 2027
- 💼 GitHub: [UchiaObito004](https://github.com/UchiaObito004)

---

## 📄 License

[MIT](LICENSE)
