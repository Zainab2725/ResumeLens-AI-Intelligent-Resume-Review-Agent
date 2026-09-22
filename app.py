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
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "groq/openai/gpt-oss-120b"


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):
    """
    Extract text from an uploaded PDF.

    Returns:
        text, error
    """

    try:
        reader = PdfReader(uploaded_file)

        if len(reader.pages) == 0:
            return None, "The uploaded PDF contains no pages."

        extracted_pages = []

        for page_number, page in enumerate(reader.pages, start=1):

            try:
                text = page.extract_text()

                if text:
                    extracted_pages.append(text)

            except Exception:
                # Continue if one page cannot be read
                continue

        final_text = "\n".join(
            extracted_pages
        ).strip()

        if not final_text:

            return None, (
                "No readable text was found in this PDF. "
                "The file may be scanned/image-based. "
                "Please upload a text-based PDF or paste your resume."
            )

        return final_text, None

    except Exception as e:

        return None, (
            f"PDF extraction failed: {str(e)}"
        )


# ============================================================
# GROQ / CREWAI LLM
# ============================================================

def create_llm():
    """
    Create the CrewAI LLM.

    The Groq API key is retrieved ONLY from
    Streamlit Secrets.
    """

    try:

        api_key = st.secrets["GROQ_API_KEY"]

    except KeyError:

        return None, (
            "GROQ_API_KEY is not configured. "
            "Please add GROQ_API_KEY in "
            "Streamlit Cloud → Settings → Secrets."
        )

    except Exception as e:

        return None, (
            f"Could not access Streamlit Secrets: {str(e)}"
        )

    if not api_key or not str(api_key).strip():

        return None, (
            "GROQ_API_KEY is empty. "
            "Please add a valid Groq API key in Streamlit Secrets."
        )

    try:

        llm = LLM(
            model=MODEL_NAME,
            api_key=api_key,
            temperature=0.1
        )

        return llm, None

    except Exception as e:

        return None, (
            f"Could not initialize the Groq model: {str(e)}"
        )


# ============================================================
# CLEAN AI JSON
# ============================================================

def clean_json_response(response):
    """
    Remove accidental Markdown code fences
    from the model response.
    """

    text = str(response).strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


# ============================================================
# RESUME REVIEW AGENT
# ============================================================

def analyze_resume(resume_text, job_description):

    llm, error = create_llm()

    if error:
        return None, error

    # --------------------------------------------------------
    # SINGLE CREWAI AGENT
    # --------------------------------------------------------

    reviewer_agent = Agent(

        role="Evidence-Based Resume Review Specialist",

        goal=(
            "Accurately analyze a candidate's resume against "
            "a target job description and provide structured, "
            "honest and actionable recommendations."
        ),

        backstory=(
            "You are an expert resume reviewer and recruitment "
            "analyst. You compare job requirements against "
            "explicit evidence in a candidate's resume. "
            "You never invent qualifications, skills, experience, "
            "education, certifications, projects, technologies, "
            "achievements, or years of experience."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False
    )

    # --------------------------------------------------------
    # TASK
    # --------------------------------------------------------

    review_task = Task(

        description=f"""

Analyze the candidate's resume against the target job description.

============================================================
CANDIDATE RESUME
============================================================

{resume_text}

============================================================
TARGET JOB DESCRIPTION
============================================================

{job_description}

============================================================
CORE RULE: NO FABRICATION
============================================================

The resume is the ONLY source of truth about the candidate.

Never assume that the candidate possesses a qualification.

For every job requirement:

MATCHED:
Use this only when the resume contains clear evidence.

PARTIAL:
Use this when the resume contains related but incomplete evidence.

NOT_MENTIONED:
Use this when the resume contains no evidence.

IMPORTANT:

"Not mentioned" does NOT mean that the candidate does not
possess the skill.

It only means the provided resume does not provide evidence
of that skill.

Never convert missing information into a negative fact.

============================================================
ANALYZE THESE AREAS
============================================================

1. Job requirements
2. Required skills
3. Preferred skills
4. Experience requirements
5. Education requirements
6. Responsibilities
7. Important keywords
8. Resume evidence
9. Missing information
10. ATS/readability issues
11. Improvement opportunities
12. Interview preparation

============================================================
OUTPUT
============================================================

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

============================================================
VALID VALUES
============================================================

category must be one of:

"skill"
"experience"
"education"
"responsibility"
"keyword"

importance must be:

"required"
"preferred"

status must be:

"matched"
"partial"
"not_mentioned"

priority must be:

"high"
"medium"
"low"

============================================================
RECOMMENDATION RULE
============================================================

Recommendations must be honest.

Never tell the candidate to falsely add experience.

BAD:

"Add AWS experience."

GOOD:

"If you have genuinely used AWS in a project or work
experience, consider adding that experience with specific
details."

BAD:

"Add 3 years of Python experience."

GOOD:

"If you have additional Python experience that is not
currently described, consider adding the relevant projects
or responsibilities."

============================================================
ATS ANALYSIS
============================================================

Identify:

- Job keywords already supported by the resume
- Important job keywords not mentioned in the resume
- Structure problems
- Readability problems

Do not claim that a resume is guaranteed to pass or fail
an ATS.

============================================================
INTERVIEW QUESTIONS
============================================================

Generate useful interview questions based on:

- Skills explicitly shown in the resume
- Job responsibilities
- Job requirements
- Areas where resume evidence is limited

Never assume the candidate has experience that is not shown.
""",

        expected_output=(
            "A valid JSON object containing a structured "
            "resume review."
        ),

        agent=reviewer_agent
    )

    # --------------------------------------------------------
    # CREW
    # --------------------------------------------------------

    crew = Crew(

        agents=[reviewer_agent],

        tasks=[review_task],

        process=Process.sequential,

        verbose=False
    )

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    try:

        result = crew.kickoff()

        cleaned_response = clean_json_response(result)

        data = json.loads(cleaned_response)

        return data, None

    except json.JSONDecodeError:

        return None, (
            "The AI returned an unexpected format. "
            "Please click Analyze again."
        )

    except Exception as e:

        message = str(e).lower()

        # Rate limit
        if "429" in message or "rate limit" in message:

            return None, (
                "⚠️ Groq rate limit reached. "
                "Please wait a little and try again."
            )

        # Authentication
        if (
            "401" in message
            or "unauthorized" in message
            or "authentication" in message
        ):

            return None, (
                "🔑 Groq authentication failed. "
                "Please check your GROQ_API_KEY in "
                "Streamlit Secrets."
            )

        # Timeout
        if "timeout" in message:

            return None, (
                "⏱️ The AI request timed out. "
                "Please try again or use a shorter resume "
                "and job description."
            )

        # Generic error
        return None, (
            f"❌ Resume analysis failed: {str(e)}"
        )


# ============================================================
# HEADER
# ============================================================

st.title("📄 ResumeLens AI")

st.subheader(
    "Evidence-Based Resume Review Agent"
)

st.write(
    "Compare your resume with a target job description "
    "and discover matching requirements, gaps, ATS issues, "
    "and honest ways to improve your resume."
)

st.info(
    "🛡️ ResumeLens AI does not invent qualifications. "
    "If something is absent from the resume, it is reported "
    "as 'Not mentioned' rather than assumed."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 ResumeLens AI")

    st.write(
        """
        **Powered by**

        • CrewAI  
        • Groq  
        • GPT-OSS 120B  
        • Streamlit  
        • PyPDF
        """
    )

    st.divider()

    st.write(
        """
        **What it checks**

        ✅ Requirement matching

        🔍 Resume evidence

        ⚠️ Missing information

        🤖 ATS-related issues

        💡 Improvement recommendations

        🎤 Interview questions
        """
    )

    st.divider()

    st.caption(
        "AI-generated analysis. "
        "Always verify recommendations before editing your resume."
    )


# ============================================================
# RESUME SECTION
# ============================================================

st.header("1️⃣ Provide Your Resume")

input_method = st.radio(
    "Choose your resume input:",
    [
        "Paste Resume Text",
        "Upload PDF"
    ],
    horizontal=True
)

resume_text = ""


# ------------------------------------------------------------
# PASTE TEXT
# ------------------------------------------------------------

if input_method == "Paste Resume Text":

    resume_text = st.text_area(
        "Paste your resume below",
        height=350,
        placeholder=(
            "Paste your complete resume here..."
        )
    )


# ------------------------------------------------------------
# PDF
# ------------------------------------------------------------

else:

    uploaded_pdf = st.file_uploader(
        "Upload your resume PDF",
        type=["pdf"],
        help="Upload a text-based PDF resume."
    )

    if uploaded_pdf:

        with st.spinner(
            "📄 Reading your resume..."
        ):

            resume_text, pdf_error = extract_pdf_text(
                uploaded_pdf
            )

        if pdf_error:

            st.error(pdf_error)

        else:

            st.success(
                "Resume successfully extracted."
            )

            with st.expander(
                "👀 Preview extracted resume"
            ):

                st.text(
                    resume_text[:10000]
                )


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.header("2️⃣ Target Job Description")

job_description = st.text_area(
    "Paste the job description",
    height=350,
    placeholder=(
        "Paste the complete job description here..."
    )
)


# ============================================================
# ANALYZE
# ============================================================

st.header("3️⃣ Resume Analysis")

analyze_button = st.button(
    "🔍 Analyze My Resume",
    type="primary",
    use_container_width=True
)


if analyze_button:

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    if not resume_text.strip():

        st.warning(
            "Please provide your resume."
        )

        st.stop()

    if not job_description.strip():

        st.warning(
            "Please provide the target job description."
        )

        st.stop()

    if len(resume_text.strip()) < 100:

        st.warning(
            "The resume appears too short. "
            "Please provide more complete resume information."
        )

        st.stop()

    if len(job_description.strip()) < 100:

        st.warning(
            "The job description appears too short. "
            "Please paste the complete job description."
        )

        st.stop()

    # --------------------------------------------------------
    # AI ANALYSIS
    # --------------------------------------------------------

    with st.spinner(
        "🤖 ResumeLens is analyzing your resume..."
    ):

        result, error = analyze_resume(
            resume_text,
            job_description
        )

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    if error:

        st.error(error)

        st.stop()

    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    st.session_state["review_result"] = result


# ============================================================
# RESULTS
# ============================================================

if "review_result" in st.session_state:

    result = st.session_state["review_result"]

    st.success(
        "✅ Resume analysis completed!"
    )

    # ========================================================
    # OVERALL REVIEW
    # ========================================================

    st.header("📊 Overall Review")

    summary = result.get(
        "summary",
        {}
    )

    st.write(
        summary.get(
            "overall_assessment",
            "No overall assessment was returned."
        )
    )

    # --------------------------------------------------------
    # STRENGTHS & GAPS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("✅ Key Strengths")

        strengths = summary.get(
            "key_strengths",
            []
        )

        if strengths:

            for strength in strengths:

                st.write(
                    f"• {strength}"
                )

        else:

            st.write(
                "No strengths were identified."
            )

    with col2:

        st.subheader("⚠️ Key Gaps")

        gaps = summary.get(
            "key_gaps",
            []
        )

        if gaps:

            for gap in gaps:

                st.write(
                    f"• {gap}"
                )

        else:

            st.write(
                "No major gaps were identified."
            )

    # ========================================================
    # REQUIREMENT MATCHING
    # ========================================================

    st.header("🎯 Requirement-by-Requirement Matching")

    st.caption(
        "Not mentioned means there is no evidence in the "
        "provided resume. It does not mean the candidate "
        "does not possess the skill."
    )

    requirements = result.get(
        "requirements",
        []
    )

    if requirements:

        for requirement in requirements:

            status = requirement.get(
                "status",
                "not_mentioned"
            )

            if status == "matched":

                icon = "🟢"

            elif status == "partial":

                icon = "🟡"

            else:

                icon = "🔴"

            title = requirement.get(
                "requirement",
                "Requirement"
            )

            with st.expander(
                f"{icon} {title}"
            ):

                st.write(
                    f"**Category:** "
                    f"{requirement.get('category', 'N/A')}"
                )

                st.write(
                    f"**Importance:** "
                    f"{requirement.get('importance', 'N/A')}"
                )

                st.write(
                    f"**Status:** "
                    f"{status.replace('_', ' ').title()}"
                )

                st.markdown(
                    "**Evidence from Resume**"
                )

                evidence = requirement.get(
                    "resume_evidence",
                    ""
                )

                if evidence:

                    st.info(evidence)

                else:

                    st.info(
                        "Not mentioned in the provided resume."
                    )

                st.markdown(
                    "**Explanation**"
                )

                st.write(
                    requirement.get(
                        "explanation",
                        ""
                    )
                )

    else:

        st.info(
            "No individual requirements were returned."
        )

    # ========================================================
    # ATS ANALYSIS
    # ========================================================

    st.header("🤖 ATS & Resume Analysis")

    ats = result.get(
        "ats_analysis",
        {}
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "✅ Supported Keywords"
        )

        supported = ats.get(
            "supported_keywords",
            []
        )

        if supported:

            for keyword in supported:

                st.write(
                    f"• {keyword}"
                )

        else:

            st.write(
                "No supported keywords identified."
            )

        st.subheader(
            "⚠️ Missing / Unmentioned Keywords"
        )

        missing = ats.get(
            "missing_keywords",
            []
        )

        if missing:

            for keyword in missing:

                st.write(
                    f"• {keyword}"
                )

        else:

            st.write(
                "No missing keywords identified."
            )

    with col2:

        st.subheader(
            "📄 Structure Issues"
        )

        structure = ats.get(
            "structure_issues",
            []
        )

        if structure:

            for issue in structure:

                st.write(
                    f"• {issue}"
                )

        else:

            st.write(
                "No major structure issues identified."
            )

        st.subheader(
            "👀 Readability Issues"
        )

        readability = ats.get(
            "readability_issues",
            []
        )

        if readability:

            for issue in readability:

                st.write(
                    f"• {issue}"
                )

        else:

            st.write(
                "No major readability issues identified."
            )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.header("💡 Improvement Recommendations")

    recommendations = result.get(
        "recommendations",
        []
    )

    if recommendations:

        for recommendation in recommendations:

            priority = recommendation.get(
                "priority",
                "medium"
            )

            if priority == "high":

                icon = "🔴"

            elif priority == "medium":

                icon = "🟡"

            else:

                icon = "🟢"

            title = recommendation.get(
                "recommendation",
                "Recommendation"
            )

            with st.expander(
                f"{icon} {title}"
            ):

                st.write(
                    f"**Priority:** "
                    f"{priority.title()}"
                )

                st.markdown(
                    "**Why:**"
                )

                st.write(
                    recommendation.get(
                        "reason",
                        ""
                    )
                )

                st.markdown(
                    "**Honest Action:**"
                )

                st.write(
                    recommendation.get(
                        "honest_action",
                        ""
                    )
                )

    else:

        st.info(
            "No recommendations were returned."
        )

    # ========================================================
    # INTERVIEW PREPARATION
    # ========================================================

    st.header("🎤 Interview Preparation")

    questions = result.get(
        "interview_questions",
        []
    )

    if questions:

        for number, question in enumerate(
            questions,
            start=1
        ):

            st.write(
                f"**{number}. {question}**"
            )

    else:

        st.info(
            "No interview questions were generated."
        )

    # ========================================================
    # EXPORT
    # ========================================================

    st.header("📥 Export Analysis")

    json_data = json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )

    st.download_button(
        label="Download Analysis JSON",
        data=json_data,
        file_name="resumelens_analysis.json",
        mime="application/json",
        use_container_width=True
    )

    st.divider()

    st.caption(
        "ResumeLens AI is an AI-assisted review tool. "
        "It analyzes only the information supplied by the user "
        "and does not guarantee hiring outcomes."
    )
