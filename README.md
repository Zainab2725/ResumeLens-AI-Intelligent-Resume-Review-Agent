# 📄 ResumeLens AI

**AI-Powered Resume Review & Job Matching Agent**

ResumeLens AI is an intelligent resume analysis application that uses **CrewAI + Groq + GPT-OSS 120B** to compare a resume against a target job description and provide an evidence-based review.

Instead of simply giving a generic resume summary, ResumeLens AI analyzes **what the job requires, what the resume explicitly proves, what is only partially supported, and what is not mentioned**.

---

## 🚀 Features

### 📄 Resume Input
* Paste resume text directly
* Upload a PDF resume
* Extract text automatically from PDF files
* Handles text-based PDFs

### 🎯 Job Description Analysis
ResumeLens AI analyzes:
* Required skills
* Preferred skills
* Experience requirements
* Education requirements
* Job responsibilities
* Important keywords

### 🔍 Evidence-Based Matching
Each job requirement is classified as:

| Status | Meaning |
| :--- | :--- |
| ✅ Matched | Resume explicitly provides evidence |
| 🟡 Partial | Resume contains related but incomplete evidence |
| ⚪ Not Mentioned | No evidence was found in the provided resume |

> **Important:** "Not Mentioned" does not mean the candidate does not have the skill. It only means that the provided resume does not contain evidence for it.

### 🤖 AI-Powered Review
The AI provides:
* Overall resume assessment
* Key strengths
* Key gaps
* Requirement-by-requirement analysis
* Resume evidence
* ATS keyword analysis
* Structure and readability issues
* Improvement recommendations
* Interview questions

### 🛡️ Anti-Fabrication Approach
ResumeLens AI follows an evidence-first approach. It does **not** invent:
* Skills
* Work experience
* Education
* Certifications
* Projects
* Achievements

Recommendations focus on improving how existing experience is presented and identifying information that may need to be added **only if it is actually true**.

---

## 🧠 How It Works

```text
Resume / PDF
     │
     ▼
Resume Text Extraction
     │
     ▼
Target Job Description
     │
     ▼
CrewAI Resume Review Agent
     │
     ▼
GPT-OSS 120B via Groq
     │
     ▼
Structured JSON Analysis
     │
     ├── Resume Summary
     ├── Requirement Matching
     ├── ATS Analysis
     ├── Recommendations
     └── Interview Questions
     │
     ▼
Interactive Streamlit Results
