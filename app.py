import json
import re

import streamlit as st
from pypdf import PdfReader
from crewai import Agent, Task, Crew, Process, LLM


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ResumeLens AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL STYLING
# ============================================================

st.markdown(
    """
    <style>
        /* Main application */
        .stApp {
            background-color: #f7f8fa;
        }

        /* Main content width */
        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Remove unnecessary Streamlit decoration */
        [data-testid="stDecoration"] {
            display: none;
        }

        /* Header */
        .app-header {
            padding: 1rem 0 2rem 0;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 2rem;
        }

        .app-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }

        .app-subtitle {
            font-size: 1rem;
            color: #6b7280;
            line-height: 1.6;
        }

        /* Section headings */
        .section-title {
            font-size: 1.25rem;
            font-weight: 650;
            color: #111827;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }

        .section-description {
            color: #6b7280;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        /* Cards */
        .metric-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 1.25rem;
            min-height: 130px;
        }

        .metric-label {
            font-size: 0.82rem;
            color: #6b7280;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #111827;
        }

        /* Status badges */
        .status-matched {
            display: inline-block;
            background: #ecfdf3;
            color: #166534;
            border: 1px solid #bbf7d0;
            padding: 3px 9px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .status-partial {
            display: inline-block;
            background: #fffbeb;
            color: #92400e;
            border: 1px solid #fde68a;
            padding: 3px 9px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .status-not-mentioned {
            display: inline-block;
            background: #f3f4f6;
            color: #4b5563;
            border: 1px solid #d1d5db;
            padding: 3px 9px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        /* Recommendation priority */
        .priority-high {
            color: #991b1b;
            font-weight: 650;
        }

        .priority-medium {
            color: #92400e;
            font-weight: 650;
        }

        .priority-low {
            color: #374151;
            font-weight: 650;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e5e7eb;
        }

        .sidebar-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.2rem;
        }

        .sidebar-subtitle {
            font-size: 0.82rem;
            color: #6b7280;
            line-height: 1.5;
            margin-bottom: 1.5rem;
        }

        .sidebar-item {
            padding: 0.65rem 0;
            border-bottom: 1px solid #f0f1f3;
        }

        .sidebar-label {
            font-size: 0.72rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .sidebar-value {
            font-size: 0.88rem;
            color: #374151;
            margin-top: 0.15rem;
        }

        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 7px;
            border: 1px solid #111827;
            background-color: #111827;
            color: white;
            font-weight: 600;
            min-height: 2.7rem;
        }

        .stButton > button:hover {
            background-color: #374151;
            border-color: #374151;
            color: white;
        }

        /* Download button */
        .stDownloadButton > button {
            border-radius: 7px;
            font-weight: 600;
        }

        /* Text inputs */
        .stTextArea textarea,
        .stTextInput input {
            border-radius: 7px;
            border: 1px solid #d1d5db;
        }

        /* Expanders */
        [data-testid="stExpander"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: white;
        }

        /* Tables */
        .dataframe {
            font-size: 0.88rem;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #9ca3af;
            font-size: 0.75rem;
            padding-top: 3rem;
            border-top: 1px solid #e5e7eb;
            margin-top: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_NAME = "groq/openai/gpt-oss-120b"


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):
    """Extract text from a PDF file."""

    try:
        reader = PdfReader(uploaded_file)

        if not reader.pages:
            return None, "The uploaded PDF does not contain any pages."

        extracted_pages = []

        for page in reader.pages:
            try:
                text = page.extract_text() or ""
                extracted_pages.append(text)
            except Exception:
                continue

        text = "\n".join(extracted_pages).strip()

        if not text:
            return None, (
                "No readable text was found in this PDF. "
                "The file may be scanned or image-based. "
                "Please upload a text-based PDF or paste the resume text."
            )

        return text, None

    except Exception:
        return None, (
            "The PDF could not be processed. "
            "Please check that the file is a valid PDF."
        )


# ============================================================
# LLM CONFIGURATION
# ============================================================

def create_llm():
    """Create the CrewAI LLM using the Streamlit secret."""

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add it under Streamlit Cloud Secrets."
        )

    if not api_key or not str(api_key).strip():
        raise RuntimeError(
            "GROQ_API_KEY is empty. "
            "Please add a valid Groq API key to Streamlit Secrets."
        )

    return LLM(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.1,
    )


# ============================================================
# JSON CLEANING
# ============================================================

def clean_json_response(response):
    """Convert the LLM response into clean JSON."""

    if hasattr(response, "raw"):
        response = response.raw

    response = str(response).strip()

    # Remove Markdown JSON fences
    response = re.sub(r"^```json\s*", "", response, flags=re.IGNORECASE)
    response = re.sub(r"^```\s*", "", response)
    response = re.sub(r"\s*```$", "", response)

    # Find JSON object if extra text was returned
    start = response.find("{")
    end = response.rfind("}")

    if start != -1 and end != -1:
        response = response[start:end + 1]

    return json.loads(response)


# ============================================================
# RESUME ANALYSIS
# ============================================================

def analyze_resume(resume_text, job_description):
    """Run the single CrewAI resume review agent."""

    llm = create_llm()

    agent = Agent(
        role="Evidence-Based Resume Review Specialist",
        goal=(
            "Analyze a resume against a target job description using "
            "only evidence explicitly available in the resume. "
            "Produce an accurate, structured and actionable review "
            "without inventing qualifications."
        ),
        backstory=(
            "You are a professional resume and recruitment analyst. "
            "You carefully compare candidate evidence with job requirements. "
            "You never assume that a candidate possesses a skill merely "
            "because it would be useful for the job. "
            "You distinguish matched evidence, partial evidence, and "
            "information that is simply not mentioned."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    task_description = f"""
Analyze the following resume against the target job description.

RESUME:
----------------
{resume_text}
----------------

JOB DESCRIPTION:
----------------
{job_description}
----------------

IMPORTANT RULES:

1. The resume is the only source of candidate evidence.
2. Never invent skills, qualifications, experience, certifications,
   projects, achievements or education.
3. "matched" means there is explicit supporting evidence in the resume.
4. "partial" means the resume contains related evidence, but it does
   not fully satisfy the requirement.
5. "not_mentioned" means no supporting evidence was found in the resume.
6. "not_mentioned" does NOT mean the candidate does not possess the skill.
7. Be specific when quoting or describing resume evidence.
8. Separate required and preferred job requirements.
9. Identify important keywords from the job description.
10. Give practical recommendations that the candidate can honestly act on.
11. Do not recommend adding a skill unless the candidate genuinely has it.
12. Analyze ATS readability and resume structure.
13. Generate useful interview questions based on the job requirements.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "summary": {{
        "overall_assessment": "",
        "key_strengths": [],
        "key_gaps": []
    }},

    "requirements": [
        {{
            "requirement": "",
            "category": "",
            "importance": "",
            "status": "",
            "resume_evidence": "",
            "explanation": ""
        }}
    ],

    "ats_analysis": {{
        "supported_keywords": [],
        "missing_keywords": [],
        "structure_issues": [],
        "readability_issues": []
    }},

    "recommendations": [
        {{
            "priority": "",
            "recommendation": "",
            "reason": "",
            "honest_action": ""
        }}
    ],

    "interview_questions": []
}}

Allowed values:

category:
- skill
- experience
- education
- responsibility
- keyword

importance:
- required
- preferred

status:
- matched
- partial
- not_mentioned

priority:
- high
- medium
- low
"""

    task = Task(
        description=task_description,
        expected_output="Valid JSON only.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    try:
        result = crew.kickoff()
        return clean_json_response(result)

    except Exception as error:
        error_text = str(error)

        if "429" in error_text or "rate limit" in error_text.lower():
            raise RuntimeError(
                "Groq rate limit reached. Please wait a moment and try again."
            )

        if (
            "401" in error_text
            or "unauthorized" in error_text.lower()
            or "authentication" in error_text.lower()
        ):
            raise RuntimeError(
                "Groq authentication failed. Please check your GROQ_API_KEY."
            )

        if "timeout" in error_text.lower():
            raise RuntimeError(
                "The AI request timed out. Please try again with a shorter "
                "resume or job description."
            )

        if isinstance(error, json.JSONDecodeError):
            raise RuntimeError(
                "The AI returned an unexpected response format. "
                "Please try the analysis again."
            )

        raise RuntimeError(
            f"The resume analysis could not be completed: {error_text}"
        )


# ============================================================
# STATUS HTML
# ============================================================

def status_badge(status):
    status = str(status).lower().strip()

    if status == "matched":
        return '<span class="status-matched">MATCHED</span>'

    if status == "partial":
        return '<span class="status-partial">PARTIAL</span>'

    return '<span class="status-not-mentioned">NOT MENTIONED</span>'


def priority_class(priority):
    priority = str(priority).lower().strip()

    if priority == "high":
        return "priority-high"

    if priority == "medium":
        return "priority-medium"

    return "priority-low"


# ============================================================
# SESSION STATE
# ============================================================

if "review_result" not in st.session_state:
    st.session_state.review_result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">ResumeLens AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        "Resume review and job matching based on evidence from the "
        "candidate's resume."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-item">
            <div class="sidebar-label">Model</div>
            <div class="sidebar-value">GPT-OSS 120B</div>
        </div>

        <div class="sidebar-item">
            <div class="sidebar-label">AI Framework</div>
            <div class="sidebar-value">CrewAI</div>
        </div>

        <div class="sidebar-item">
            <div class="sidebar-label">LLM Provider</div>
            <div class="sidebar-value">Groq</div>
        </div>

        <div class="sidebar-item">
            <div class="sidebar-label">Interface</div>
            <div class="sidebar-value">Streamlit</div>
        </div>

        <div class="sidebar-item">
            <div class="sidebar-label">Analysis Approach</div>
            <div class="sidebar-value">Evidence-based review</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.caption(
        "Candidate information is evaluated only from the resume "
        "content provided to the application."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">ResumeLens AI</div>
        <div class="app-subtitle">
            Resume Review and Job Matching
        </div>
        <div class="app-subtitle" style="margin-top: 0.35rem;">
            Analyze how well a resume supports the requirements of a
            target job using evidence-based AI review.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Resume</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Provide the resume you want to evaluate."
    "</div>",
    unsafe_allow_html=True,
)

input_method = st.radio(
    "Resume input method",
    ["Paste resume text", "Upload PDF"],
    horizontal=True,
    label_visibility="collapsed",
)

resume_text = ""

if input_method == "Paste resume text":

    resume_text = st.text_area(
        "Resume text",
        height=330,
        placeholder=(
            "Paste the complete resume text here..."
        ),
        label_visibility="collapsed",
    )

else:

    uploaded_file = st.file_uploader(
        "Upload resume PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file:

        resume_text, pdf_error = extract_pdf_text(uploaded_file)

        if pdf_error:
            st.error(pdf_error)
            resume_text = ""

        if resume_text:

            with st.expander("View extracted resume text"):
                st.text_area(
                    "Extracted text",
                    resume_text,
                    height=250,
                    label_visibility="collapsed",
                )


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.markdown(
    '<div class="section-title">Target Job Description</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Paste the job description for the position you are targeting."
    "</div>",
    unsafe_allow_html=True,
)

job_description = st.text_area(
    "Job description",
    height=300,
    placeholder="Paste the complete job description here...",
    label_visibility="collapsed",
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

analyze_clicked = st.button(
    "Analyze Resume",
    use_container_width=True,
)


# ============================================================
# VALIDATION + ANALYSIS
# ============================================================

if analyze_clicked:

    if not resume_text or not resume_text.strip():
        st.error("Please provide a resume before starting the analysis.")

    elif not job_description or not job_description.strip():
        st.error(
            "Please provide a target job description before starting "
            "the analysis."
        )

    else:

        # Prevent unnecessarily huge requests
        resume_text = resume_text.strip()
        job_description = job_description.strip()

        if len(resume_text) > 50000:
            st.warning(
                "The resume is very long. Only the first 50,000 characters "
                "will be analyzed."
            )
            resume_text = resume_text[:50000]

        if len(job_description) > 40000:
            st.warning(
                "The job description is very long. Only the first 40,000 "
                "characters will be analyzed."
            )
            job_description = job_description[:40000]

        with st.spinner("Analyzing resume against the job description..."):

            try:

                result = analyze_resume(
                    resume_text,
                    job_description,
                )

                st.session_state.review_result = result

                st.success("Resume analysis completed.")

            except Exception as error:

                st.session_state.review_result = None

                st.error(str(error))


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.review_result

if result:

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Review Summary</div>',
        unsafe_allow_html=True,
    )

    summary = result.get("summary", {})

    overall_assessment = summary.get(
        "overall_assessment",
        "No overall assessment was generated.",
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Overall Assessment</div>
            <div style="
                color:#374151;
                line-height:1.7;
                font-size:0.95rem;
            ">
                {overall_assessment}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="section-title">Key Strengths</div>',
            unsafe_allow_html=True,
        )

        strengths = summary.get("key_strengths", [])

        if strengths:

            for strength in strengths:
                st.markdown(f"- {strength}")

        else:
            st.caption("No strengths were identified.")

    with col2:

        st.markdown(
            '<div class="section-title">Key Gaps</div>',
            unsafe_allow_html=True,
        )

        gaps = summary.get("key_gaps", [])

        if gaps:

            for gap in gaps:
                st.markdown(f"- {gap}")

        else:
            st.caption("No major gaps were identified.")


    # ========================================================
    # REQUIREMENT MATCHING
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Requirement Matching</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        "Each requirement is evaluated against explicit evidence in "
        "the resume."
        "</div>",
        unsafe_allow_html=True,
    )

    requirements = result.get("requirements", [])

    if requirements:

        for index, requirement in enumerate(requirements, start=1):

            req_name = requirement.get(
                "requirement",
                "Unnamed requirement",
            )

            category = requirement.get(
                "category",
                "Not specified",
            )

            importance = requirement.get(
                "importance",
                "Not specified",
            )

            status = requirement.get(
                "status",
                "not_mentioned",
            )

            evidence = requirement.get(
                "resume_evidence",
                "No evidence provided.",
            )

            explanation = requirement.get(
                "explanation",
                "",
            )

            with st.container(border=True):

                left, right = st.columns([4, 1])

                with left:

                    st.markdown(
                        f"**{index}. {req_name}**"
                    )

                    st.caption(
                        f"Category: {category} | "
                        f"Importance: {importance}"
                    )

                with right:

                    st.markdown(
                        status_badge(status),
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"**Resume evidence:** {evidence}"
                )

                if explanation:

                    st.markdown(
                        f"**Analysis:** {explanation}"
                    )

    else:

        st.info(
            "No requirement-level analysis was returned."
        )


    # ========================================================
    # ATS ANALYSIS
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">ATS and Resume Analysis</div>',
        unsafe_allow_html=True,
    )

    ats = result.get("ats_analysis", {})

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Supported Keywords**")

        supported = ats.get(
            "supported_keywords",
            [],
        )

        if supported:

            for keyword in supported:
                st.markdown(f"- {keyword}")

        else:

            st.caption("No supported keywords identified.")

        st.markdown("**Missing Keywords**")

        missing = ats.get(
            "missing_keywords",
            [],
        )

        if missing:

            for keyword in missing:
                st.markdown(f"- {keyword}")

        else:

            st.caption("No missing keywords identified.")

    with col2:

        st.markdown("**Structure Issues**")

        structure = ats.get(
            "structure_issues",
            [],
        )

        if structure:

            for issue in structure:
                st.markdown(f"- {issue}")

        else:

            st.caption("No major structure issues identified.")

        st.markdown("**Readability Issues**")

        readability = ats.get(
            "readability_issues",
            [],
        )

        if readability:

            for issue in readability:
                st.markdown(f"- {issue}")

        else:

            st.caption("No major readability issues identified.")


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Improvement Recommendations</div>',
        unsafe_allow_html=True,
    )

    recommendations = result.get(
        "recommendations",
        [],
    )

    if recommendations:

        for recommendation in recommendations:

            priority = recommendation.get(
                "priority",
                "low",
            )

            recommendation_text = recommendation.get(
                "recommendation",
                "",
            )

            reason = recommendation.get(
                "reason",
                "",
            )

            honest_action = recommendation.get(
                "honest_action",
                "",
            )

            with st.container(border=True):

                st.markdown(
                    f'<div class="{priority_class(priority)}">'
                    f'{priority.upper()} PRIORITY'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"**{recommendation_text}**"
                )

                if reason:

                    st.markdown(
                        f"**Why:** {reason}"
                    )

                if honest_action:

                    st.markdown(
                        f"**Recommended action:** {honest_action}"
                    )

    else:

        st.caption(
            "No improvement recommendations were generated."
        )


    # ========================================================
    # INTERVIEW PREPARATION
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Interview Preparation</div>',
        unsafe_allow_html=True,
    )

    interview_questions = result.get(
        "interview_questions",
        [],
    )

    if interview_questions:

        for index, question in enumerate(
            interview_questions,
            start=1,
        ):

            st.markdown(
                f"**{index}. {question}**"
            )

    else:

        st.caption(
            "No interview questions were generated."
        )


    # ========================================================
    # JSON EXPORT
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Export Analysis</div>',
        unsafe_allow_html=True,
    )

    json_data = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

    st.download_button(
        label="Download JSON Report",
        data=json_data,
        file_name="resume_review_report.json",
        mime="application/json",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ResumeLens AI — Evidence-based resume review and job matching
    </div>
    """,
    unsafe_allow_html=True,
)
````

### Also add `runtime.txt`

Because your current Streamlit deployment is using **Python 3.14**, add this file:

python-3.12

### `requirements.txt`

Use:

streamlit
crewai
groq
pypdf

So your GitHub repository becomes:

```text
resume-lens-ai/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

**Important:** after uploading `runtime.txt`, redeploy/reboot the Streamlit app. The previous error was happening during the CrewAI → ChromaDB → Pydantic import, before ResumeLens AI even reached Groq.
