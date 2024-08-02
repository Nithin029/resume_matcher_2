from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import pandas as pd
import fitz
from typing import List,Tuple
import streamlit as st
import numpy as np
import tiktoken
import tkinter as tk
from tkinter import filedialog, simpledialog
import re
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HELICONE_API_KEY=os.getenv("HELICONE_API_KEY")
SysPromptDefault = "You are now in the role of an expert AI."

def response(message: object, model: object = "llama3-8b-8192", SysPrompt: object = SysPromptDefault, temperature: object = 0.2) -> object:
    """

    :rtype: object
    """
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://gateway.hconeai.com/openai/v1",
        default_headers={
            "Helicone-Auth": f"Bearer {HELICONE_API_KEY}",
            "Helicone-Target-Url": "https://api.groq.com"
        }
    )

    messages = [{"role": "system", "content": SysPrompt}, {"role": "user", "content": message}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        frequency_penalty=0.2,
    )
    return response.choices[0].message.content

def extract_content(pdf_content: bytes) -> List[str]:
    """
    Takes PDF (bytes) and returns a list of strings containing text from each page.
    """
    pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")

    pages_content = []
    for page_number in range(pdf_doc.page_count):
        # Extracting text content
        page = pdf_doc.load_page(page_number)
        text_content = page.get_text("text").replace("\n", "\t")
        pages_content.append(text_content)

    pdf_doc.close()
    return pages_content

resume_prompt="""

**You are a professional resume analyst with extensive experience in parsing and evaluating candidate resumes for recruitment purposes. Your task is to analyze a given candidate's resume and extract specific information related to their recent qualifications, skills, work experience, and total professional experience. You have a keen eye for detail and a thorough understanding of how to present information in a structured format.**

**Objective:** Extract and structure relevant data from the RESUME in JSON format, focusing on the most recent qualifications, skills, work experience, and total professional experience.

**Instructions:**

1. **Qualification:**
   - Extract the most recent qualification of the candidate.
   - Ensure it is limited to professional certifications or advanced degrees (exclude schools and colleges).
   - Provide only the most recent qualification.

2. **Skills:**
   - List the candidate’s key skills relevant to their profession.
   - Include technical skills, soft skills, and any other professional competencies mentioned.

3. **Recent Experience:**
   - Identify and list the candidate’s most recent job experience.
   - For the recent experience, include:
     - **Role**: The job title or position held.
     - **Company**: The name of the company where the experience was gained.
     - **Experience Duration**: The total duration of the recent experience in years (in numerical format).
   - Summarize the projects undertaken in this role in 2 to 3 lines, highlighting the primary responsibilities and outcomes.

4. **Total Experience:**
   - Calculate the candidate’s total professional experience in years.
   - This includes all relevant work experiences, adding up the duration from each job listed on the resume.

5. **Output Format:**
   - Structure the extracted information in a JSON format with the following keys:
     - `recent_qualification`
     - `skills`
     - `recent_experience` which includes `role`, `company`, `experience_duration`, and `projects_summary`.
     - `total_experience`

**Sample JSON Output:**
```json
{
  "recent_qualification": "Project Management Professional (PMP) Certification",
  "skills": ["JavaScript", "Project Management", "Team Leadership", "Agile Methodologies", "Problem Solving"],
  "recent_experience": {
    "role": "Senior Project Manager",
    "company": "Tech Solutions Inc.",
    "experience_duration": 3,
    "projects_summary": "Managed a team of developers to deliver a web-based application, leading to a 20% increase in client satisfaction. Successfully implemented agile processes that improved project turnaround time by 15%."
  },
  "total_experience": 10
}
```

**Steps:**
1. Parse the resume and identify sections relevant to qualifications, skills, and work experience.
2. Extract the most recent professional qualification.
3. Compile a list of key skills mentioned.
4. Determine the most recent job experience, including the role, company, and duration.
5. Summarize the key projects and responsibilities undertaken in the recent role.
6. Calculate the total professional experience by adding up the duration of each job listed in the resume.
7. Format the extracted data into a structured JSON object as specified.

Take a deep breath and work on this problem step-by-step.
"""

jd_prompt="""

**You are an HR data analyst with expertise in extracting structured information from JOB_DESCRIPTION. You specialize in identifying key details such as job roles, requirements, qualifications, experience, and skill sets in a precise and systematic manner.**

**Objective:** Your task is to analyze the provided JOB_DESCRIPTION and extract the following details:
1. **Position/Role:** The name of the job position or role being advertised.
2. **Requirements for the Role:** Key responsibilities and tasks associated with the job.
3. **Qualifications:** The required educational background or certifications.
4. **Experience Required:** The number of years of experience required for the job, in numerical format.
5. **Skill Set:** The specific skills required for the job, including both technical and soft skills.

**Steps:**
1. **Identify the Position/Role:** Locate the title or primary name of the job being described. Extract this information as a string.
2. **Determine the Requirements for the Role:** List the main responsibilities and tasks expected from the candidate in a detailed manner.
3. **Extract Qualifications:** Identify the required educational background and certifications. Specify if any particular degree or certification is needed.
4. **Calculate the Experience Required:** Extract the number of years of professional experience required. If a range is provided, list the minimum and maximum years. If only a single value is mentioned, note that value.
5. **List the Skill Set:** Identify and list both technical and soft skills required for the role. Technical skills may include specific software or tools, while soft skills may include communication or teamwork abilities.

**Format the Output:** Return the extracted information in valid JSON format as shown below:

```json
{
  "position_role": "string",
  "requirements": [
    "string",
    "string",
    ...
  ],
  "qualifications": [
    "string",
    "string",
    ...
  ],
  "experience_required": {
    "min_years": number,
    "max_years": number
  },
  "skill_set": {
    "technical_skills": [
      "string",
      "string",
      ...
    ],
    "soft_skills": [
      "string",
      "string",
      ...
    ]
  }
}
```

**Example Output:**

```json
{
  "position_role": "Software Engineer",
  "requirements": [
    "Develop and maintain software applications",
    "Collaborate with cross-functional teams",
    "Write clean, scalable code"
  ],
  "qualifications": [
    "Bachelor's degree in Computer Science",
    "Proficiency in Java and Python"
  ],
  "experience_required": {
    "min_years": 2,
    "max_years": 5
  },
  "skill_set": {
    "technical_skills": [
      "Java",
      "Python",
      "SQL"
    ],
    "soft_skills": [
      "Team collaboration",
      "Problem-solving",
      "Effective communication"
    ]
  }
}
```

Please ensure all fields are accurately filled based on the job description. If any field is not applicable, leave it empty but maintain the JSON structure. If any value is indeterminate, use `null` for numbers and an empty string for strings.

**Take a deep breath and work on this problem step-by-step.**
"""

grading_prompt="""

**You are an expert in human resources and data analysis with over 15 years of experience in evaluating job descriptions and resumes. Your expertise includes matching job requirements with candidate qualifications and providing detailed feedback based on various parameters such as role fit, skillset, experience, and educational background.**

**Objective:**
You are to analyze a provided JOB_DESCRIPTION and a RESUME  provided by the user. Your task is to evaluate the match between the job requirements and the candidate's qualifications. The output should be in JSON format, highlighting the weightage, score, and reasoning for each category: role, skillset, experience, and educational qualifications.
Be conservative in assigning the score.
**Steps:**

1. **Input Texts:**
   - Extract the provided job description and resume from the input text.
   - Delimit each part using triple backticks for clarity.

2. **Identify Job Description Components:**
   - Parse the job description to extract:
     - **Role:** Identify the job title or primary role being advertised.
     - **Skillset:** List the specific skills required for the job, including technical, soft, and any specific software or tools.
     - **Experience:** Mention the years of experience or specific types of experience required.
     - **Educational Qualifications:** Note any required or preferred educational background, such as degrees or certifications.

3. **Identify Resume Components:**
   - Parse the resume to extract:
     - **Role:** Identify the current or most recent job title and any relevant past job titles.
     - **Skillset:** List the skills mentioned, including technical, soft, and specific software or tools.
     - **Experience:** Note the years of experience and the type of experience in various roles.
     - **Educational Qualifications:** List the educational background, including degrees and certifications.

4. **Matching Criteria:**
   - Compare each category from the job description with the corresponding category from the resume.
   - Evaluate the match and assign a weightage (e.g., 1 to 10) for each category based on relevance and completeness of match.
   - Provide a score (e.g., percentage) that reflects how well the candidate’s qualifications meet the job requirements.Follow a conservative approach in assigning the score.
   - Offer reasoning for each score, explaining how the candidate's qualifications align or diverge from the job requirements.

5. **Generate JSON Output:**
   - Structure the output in JSON format with the following keys:
     - **role_match**: { "weightage": X, "score": Y, "reasoning": "Detailed analysis" }
     - **skillset_match**: { "weightage": X, "score": Y, "reasoning": "Detailed analysis" }
     - **experience_match**: { "weightage": X, "score": Y, "reasoning": "Detailed analysis" }
     - **education_match**: { "weightage": X, "score": Y, "reasoning": "Detailed analysis" }
   - Ensure each key has a nested structure that clearly outlines the evaluation for each category.

6. **Final Summary:**
   - Summarize the overall match between the job description and the resume.
   - Provide an overall score and a brief narrative on the suitability of the candidate for the job role.

**Example JSON Output Structure:**
```json
{
  "role_match": {
    "weightage": 8,
    "score": 90,
    "reasoning": "The candidate's current and previous job titles align well with the role specified in the job description."
  },
  "skillset_match": {
    "weightage": 7,
    "score": 85,
    "reasoning": "The candidate possesses most of the required technical skills, with some minor gaps in specific tools."
  },
  "experience_match": {
    "weightage": 9,
    "score": 95,
    "reasoning": "The candidate has the required years of experience and relevant industry background."
  },
  "education_match": {
    "weightage": 6,
    "score": 80,
    "reasoning": "The candidate's educational background meets the requirements but lacks some preferred certifications."
  },
  "overall_score": 87.5,
  "summary": "The candidate is a strong match for the job role, with relevant experience and skills. The educational background is adequate, though additional certifications could enhance suitability."
}
```

Take a deep breath and work on this problem step-by-step."""


def extract_json(response_str):
    """Extract the JSON part from the response string and handle comments."""
    # Remove single-line comments (//) and multi-line comments (/* */)
    response_str = re.sub(r'//.*?\n|/\*.*?\*/', '', response_str, flags=re.DOTALL)

    match = re.search(r"\{.*}", response_str, re.DOTALL)
    if match:
        json_part = match.group()
        try:
            parsed_json = json.loads(json_part)  # Check if it's valid JSON
            return json_part
        except json.JSONDecodeError as e:
            print("Invalid JSON detected. Error:", e)
            print("JSON part:", json_part)
    else:
        print("No JSON part found in the response string.")
    return None

def identification(pdf_content):
    data = extract_content(pdf_content)
    print(data)
    model = "llama3-70b-8192"
    context = "\n\n".join(data)
    message = f"RESUME\n\n{context}\n\n"
    response_str = response(message=message, model=model, SysPrompt=resume_prompt, temperature=0)
    json_part = extract_json(response_str)
    return json_part


def classify_jd(job_description):
    model = "llama3-70b-8192"
    message = f"JOB_DESCRIPTION\n\n{job_description}\n\n"
    response_str = response(message=message, model=model, SysPrompt=jd_prompt, temperature=0)
    print(response_str)
    json_part = extract_json(response_str)
    return json_part

def grading(pdf_content,job_description):
    resume_str=identification(pdf_content)
    jd_str=classify_jd(job_description)
    print(jd_str)
    message = f"RESUME:\n\n{json.dumps(resume_str, indent=4)}\n\nJOB_DESCRIPTION:\n\n{json.dumps(jd_str,indent=4)}\n\n"
    model = "llama3-70b-8192"
    output_str = response(message=message, model=model, SysPrompt=grading_prompt, temperature=0)
    output=json.loads(extract_json(output_str))
    print(output)
    return output

def display_grading_results(grading_results):
    df = {
        "Criteria": [],
        "Score": [],
        "Weightage": [],
        "Reasoning": []
    }

    # Populate the dataframe
    for section, values in grading_results.items():
        if section != 'overall_score' and section != 'summary':
            df["Criteria"].append(section)
            df["Score"].append(values["score"])
            df["Weightage"].append(values["weightage"])
            df["Reasoning"].append(values["reasoning"])

    grading_df = pd.DataFrame(df)

    st.header("Grading Results")
    st.table(data=grading_df)
    st.subheader(f"Overall Score: {grading_results['overall_score']}")
    st.write(f"Summary: {grading_results['summary']}")


def main():
    st.title("Resume Matcher")

    # File uploader for PDF
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

    # Text area for job description input
    job_description = st.text_area("Enter Job Description", height=200)

    if st.button("Submit"):
        if pdf_file is not None and job_description:
            try:
                pdf_content = pdf_file.read()

                # Assuming grading function exists
                grades = grading(pdf_content, job_description)
                st.success(f"Grading completed successfully.")
                display_grading_results(grades)

            except Exception as e:
                st.error(f"An error occurred while processing the PDF: {e}")


if __name__ == "__main__":
    main()