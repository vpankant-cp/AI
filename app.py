# app.py
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import os
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import uuid
from fastapi.staticfiles import StaticFiles
import re

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = FastAPI(title="AI Talent Rediscovery API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
CANDIDATES_CSV = "data/candidates.csv"
os.makedirs("data", exist_ok=True)


# Models
class Skill(BaseModel):
    name: str
    level: int
    lastUsed: Optional[str] = None


class Experience(BaseModel):
    title: str
    company: str
    startDate: str
    endDate: Optional[str] = None
    description: Optional[str] = None
    skills: Optional[List[str]] = []


class Education(BaseModel):
    degree: str
    institution: str
    graduationYear: int
    fieldOfStudy: Optional[str] = None


class Application(BaseModel):
    position: str
    date: str
    status: str
    feedback: Optional[str] = None


class CandidateBase(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    location: Optional[str] = None
    lastPosition: Optional[str] = None
    lastApplicationDate: str
    topSkills: List[str] = []
    skills: List[Dict[str, Any]] = []
    experience: List[Dict[str, Any]] = []
    education: List[Dict[str, Any]] = []
    applications: List[Dict[str, Any]] = []
    tags: List[str] = []
    resumeText: Optional[str] = None
    totalYearsExperience: Optional[float] = None


class CandidateCreate(CandidateBase):
    pass


class CandidateDB(CandidateBase):
    id: str
    createdAt: str
    updatedAt: str


class CandidateList(BaseModel):
    id: str
    name: str
    email: str
    phone: Optional[str] = None
    location: Optional[str] = None
    lastPosition: Optional[str] = None
    lastApplicationDate: str
    topSkills: List[str] = []
    matchScore: Optional[float] = None


class JobDetails(BaseModel):
    title: str
    department: Optional[str] = None
    description: str
    requiredSkills: List[str]
    preferredSkills: List[str] = []
    yearsExperience: Optional[int] = None
    education: Optional[str] = None
    location: Optional[str] = None


# CSV Database Helper Functions
def initialize_db():
    """Initialize the CSV database if it doesn't exist"""
    if not os.path.exists(CANDIDATES_CSV):
        # Create empty dataframe with necessary columns
        df = pd.DataFrame(columns=[
            'id', 'name', 'email', 'phone', 'location', 'lastPosition',
            'lastApplicationDate', 'topSkills', 'skills', 'experience',
            'education', 'applications', 'tags', 'resumeText',
            'totalYearsExperience', 'createdAt', 'updatedAt'
        ])
        df.to_csv(CANDIDATES_CSV, index=False)

        # Add sample data for testing
        add_sample_data()


def load_candidates_df():
    """Load candidates from CSV file"""
    try:
        df = pd.read_csv(CANDIDATES_CSV, converters={
            'topSkills': parse_list_column,
            'skills': parse_json_column,
            'experience': parse_json_column,
            'education': parse_json_column,
            'applications': parse_json_column,
            'tags': parse_list_column
        })
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            'id', 'name', 'email', 'phone', 'location', 'lastPosition',
            'lastApplicationDate', 'topSkills', 'skills', 'experience',
            'education', 'applications', 'tags', 'resumeText',
            'totalYearsExperience', 'createdAt', 'updatedAt'
        ])


def save_candidates_df(df):
    """Save candidates DataFrame to CSV file"""
    df.to_csv(CANDIDATES_CSV, index=False)


def parse_list_column(x):
    """Parse list columns from CSV"""
    if pd.isna(x):
        return []
    try:
        return json.loads(x.replace("'", '"'))
    except:
        return []


def parse_json_column(x):
    """Parse JSON columns from CSV"""
    if pd.isna(x):
        return []
    try:
        return json.loads(x.replace("'", '"'))
    except:
        return []


def add_sample_data():
    """Add sample candidate data for testing purposes"""
    sample_candidates = [
        {
            "id": str(uuid.uuid4()),
            "name": "Alex Johnson",
            "email": "alex.johnson@example.com",
            "phone": "123-456-7890",
            "location": "San Francisco, CA",
            "lastPosition": "Senior Frontend Developer",
            "lastApplicationDate": "2024-01-15",
            "topSkills": ["React", "JavaScript", "TypeScript"],
            "skills": [
                {"name": "React", "level": 5, "lastUsed": "2024-01-01"},
                {"name": "JavaScript", "level": 5, "lastUsed": "2024-01-01"},
                {"name": "TypeScript", "level": 4, "lastUsed": "2024-01-01"},
                {"name": "HTML/CSS", "level": 5, "lastUsed": "2024-01-01"},
                {"name": "Node.js", "level": 3, "lastUsed": "2023-06-01"}
            ],
            "experience": [
                {
                    "title": "Senior Frontend Developer",
                    "company": "Tech Innovations Inc.",
                    "startDate": "2021-03",
                    "endDate": "2024-01",
                    "description": "Led the frontend team in developing modern web applications using React and TypeScript.",
                    "skills": ["React", "TypeScript", "Redux", "Jest"]
                },
                {
                    "title": "Frontend Developer",
                    "company": "WebSolutions Co.",
                    "startDate": "2018-05",
                    "endDate": "2021-02",
                    "description": "Developed responsive web applications and implemented new features using JavaScript frameworks.",
                    "skills": ["JavaScript", "Vue.js", "HTML", "CSS"]
                }
            ],
            "education": [
                {
                    "degree": "Bachelor of Science in Computer Science",
                    "institution": "University of California, Berkeley",
                    "graduationYear": 2018,
                    "fieldOfStudy": "Computer Science"
                }
            ],
            "applications": [
                {
                    "position": "Senior React Developer",
                    "date": "2024-01-15",
                    "status": "Screened",
                    "feedback": "Strong technical skills, good cultural fit."
                }
            ],
            "tags": ["frontend", "react", "senior"],
            "resumeText": "Experienced frontend developer with strong skills in React, JavaScript, and TypeScript. 6+ years of experience building scalable web applications.",
            "totalYearsExperience": 6,
            "createdAt": "2024-01-15T12:00:00",
            "updatedAt": "2024-01-15T12:00:00"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Maria Garcia",
            "email": "maria.garcia@example.com",
            "phone": "987-654-3210",
            "location": "Austin, TX",
            "lastPosition": "Data Scientist",
            "lastApplicationDate": "2024-02-02",
            "topSkills": ["Python", "Machine Learning", "SQL"],
            "skills": [
                {"name": "Python", "level": 5, "lastUsed": "2024-02-01"},
                {"name": "Machine Learning", "level": 4, "lastUsed": "2024-02-01"},
                {"name": "SQL", "level": 4, "lastUsed": "2024-02-01"},
                {"name": "TensorFlow", "level": 3, "lastUsed": "2023-12-01"},
                {"name": "Data Visualization", "level": 4, "lastUsed": "2024-01-15"}
            ],
            "experience": [
                {
                    "title": "Data Scientist",
                    "company": "DataTech Solutions",
                    "startDate": "2020-06",
                    "endDate": "2024-02",
                    "description": "Developed machine learning models for customer behavior prediction and data analysis.",
                    "skills": ["Python", "Machine Learning", "TensorFlow", "SQL"]
                },
                {
                    "title": "Data Analyst",
                    "company": "Analytics Corp",
                    "startDate": "2018-01",
                    "endDate": "2020-05",
                    "description": "Performed data analysis and created visualizations to support business decisions.",
                    "skills": ["SQL", "Python", "Tableau", "Excel"]
                }
            ],
            "education": [
                {
                    "degree": "Master of Science in Data Science",
                    "institution": "University of Texas at Austin",
                    "graduationYear": 2018,
                    "fieldOfStudy": "Data Science"
                },
                {
                    "degree": "Bachelor of Science in Statistics",
                    "institution": "University of Arizona",
                    "graduationYear": 2016,
                    "fieldOfStudy": "Statistics"
                }
            ],
            "applications": [
                {
                    "position": "Senior Data Scientist",
                    "date": "2024-02-02",
                    "status": "Interviewed",
                    "feedback": "Excellent technical skills and communication ability."
                }
            ],
            "tags": ["data science", "machine learning", "python"],
            "resumeText": "Experienced data scientist with a strong background in machine learning, predictive modeling, and data analysis. Skilled in Python, SQL, and various ML frameworks.",
            "totalYearsExperience": 6,
            "createdAt": "2024-02-02T09:30:00",
            "updatedAt": "2024-02-10T14:15:00"
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Jordan Smith",
            "email": "jordan.smith@example.com",
            "phone": "555-123-4567",
            "location": "New York, NY",
            "lastPosition": "DevOps Engineer",
            "lastApplicationDate": "2023-11-20",
            "topSkills": ["Kubernetes", "Docker", "AWS"],
            "skills": [
                {"name": "Kubernetes", "level": 5, "lastUsed": "2023-11-15"},
                {"name": "Docker", "level": 5, "lastUsed": "2023-11-15"},
                {"name": "AWS", "level": 4, "lastUsed": "2023-11-15"},
                {"name": "Terraform", "level": 4, "lastUsed": "2023-10-01"},
                {"name": "CI/CD", "level": 5, "lastUsed": "2023-11-10"}
            ],
            "experience": [
                {
                    "title": "DevOps Engineer",
                    "company": "Cloud Systems Inc.",
                    "startDate": "2020-01",
                    "endDate": "2023-11",
                    "description": "Implemented and managed containerized infrastructure using Kubernetes and Docker. Set up automated CI/CD pipelines.",
                    "skills": ["Kubernetes", "Docker", "AWS", "Terraform", "Jenkins"]
                },
                {
                    "title": "Systems Administrator",
                    "company": "TechOps Co.",
                    "startDate": "2017-03",
                    "endDate": "2019-12",
                    "description": "Managed Linux servers and cloud infrastructure. Automated deployment processes.",
                    "skills": ["Linux", "AWS", "Bash", "Python"]
                }
            ],
            "education": [
                {
                    "degree": "Bachelor of Science in Computer Engineering",
                    "institution": "Cornell University",
                    "graduationYear": 2017,
                    "fieldOfStudy": "Computer Engineering"
                }
            ],
            "applications": [
                {
                    "position": "Senior DevOps Engineer",
                    "date": "2023-11-20",
                    "status": "Rejected",
                    "feedback": "Good technical skills but not enough experience with our specific tech stack."
                }
            ],
            "tags": ["devops", "cloud", "infrastructure"],
            "resumeText": "DevOps engineer with extensive experience in container orchestration, infrastructure as code, and CI/CD pipelines. Skilled in Kubernetes, Docker, AWS, and Terraform.",
            "totalYearsExperience": 6.5,
            "createdAt": "2023-11-20T10:45:00",
            "updatedAt": "2023-11-25T16:30:00"
        }
    ]

    # Convert to DataFrame and save
    df = pd.DataFrame(sample_candidates)
    df['topSkills'] = df['topSkills'].apply(json.dumps)
    df['skills'] = df['skills'].apply(json.dumps)
    df['experience'] = df['experience'].apply(json.dumps)
    df['education'] = df['education'].apply(json.dumps)
    df['applications'] = df['applications'].apply(json.dumps)
    df['tags'] = df['tags'].apply(json.dumps)

    save_candidates_df(df)


# Helper functions
def preprocess_text(text):
    """Preprocess text for NLP operations"""
    if not text:
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(tokens)


def calculate_match_score(candidate, job_details):
    """Calculate match score between candidate and job"""
    score = 0
    max_score = 0

    # Required skills (50% of total score)
    if job_details.requiredSkills:
        max_score += 50
        candidate_skills = set([skill["name"].lower() for skill in candidate.get("skills", [])])
        candidate_skills.update([skill.lower() for skill in candidate.get("topSkills", [])])

        required_skills = set([skill.lower() for skill in job_details.requiredSkills])
        if required_skills:
            matched_required = len(required_skills.intersection(candidate_skills))
            score += (matched_required / len(required_skills)) * 50

    # Preferred skills (20% of total score)
    if job_details.preferredSkills:
        max_score += 20
        candidate_skills = set([skill["name"].lower() for skill in candidate.get("skills", [])])
        candidate_skills.update([skill.lower() for skill in candidate.get("topSkills", [])])

        preferred_skills = set([skill.lower() for skill in job_details.preferredSkills])
        if preferred_skills:
            matched_preferred = len(preferred_skills.intersection(candidate_skills))
            score += (matched_preferred / len(preferred_skills)) * 20

    # Experience (15% of total score)
    if job_details.yearsExperience is not None and candidate.get("totalYearsExperience") is not None:
        max_score += 15
        if candidate["totalYearsExperience"] >= job_details.yearsExperience:
            score += 15
        else:
            # Partial score
            score += (candidate["totalYearsExperience"] / job_details.yearsExperience) * 15

    # Location (5% of total score)
    if job_details.location and candidate.get("location"):
        max_score += 5
        if job_details.location.lower() in candidate["location"].lower():
            score += 5

    # Job description similarity (10% of total score)
    if job_details.description and candidate.get("resumeText"):
        max_score += 10
        vectorizer = TfidfVectorizer()
        job_desc_processed = preprocess_text(job_details.description)
        resume_processed = preprocess_text(candidate["resumeText"])

        try:
            tfidf_matrix = vectorizer.fit_transform([job_desc_processed, resume_processed])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            score += similarity * 10
        except:
            # Handle sparse matrix errors
            pass

    # Normalize score
    final_score = 0 if max_score == 0 else (score / max_score) * 100
    return round(final_score, 1)


# API Routes
@app.on_event("startup")
def startup_event():
    initialize_db()


@app.get("/api/candidates", response_model=List[CandidateList])
async def get_candidates(
        page: int = Query(1, ge=1),
        limit: int = Query(20, ge=1, le=100),
):
    df = load_candidates_df()

    # Sorting and pagination
    df = df.sort_values(by="lastApplicationDate", ascending=False)
    total = len(df)
    start_idx = (page - 1) * limit
    end_idx = min(start_idx + limit, total)

    # Handle empty dataframe
    if df.empty:
        return []

    # Convert to list of dictionaries
    candidates = []
    for idx, row in df.iloc[start_idx:end_idx].iterrows():
        candidate_dict = row.to_dict()
        candidates.append(CandidateList(**candidate_dict))

    return candidates


@app.get("/api/candidates/{candidate_id}", response_model=CandidateDB)
async def get_candidate(candidate_id: str):
    df = load_candidates_df()

    # Find candidate by ID
    candidate = df[df['id'] == candidate_id]
    if candidate.empty:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Convert to dictionary and return
    candidate_dict = candidate.iloc[0].to_dict()
    return CandidateDB(**candidate_dict)


@app.get("/api/candidates/search", response_model=List[CandidateList])
async def search_candidates(
        q: Optional[str] = None,
        skills: Optional[str] = None,
        experience: Optional[int] = None,
        location: Optional[str] = None,
        applied_since: Optional[str] = None,
        page: int = Query(1, ge=1),
        limit: int = Query(20, ge=1, le=100),
):
    df = load_candidates_df()

    # Apply filters
    if df.empty:
        return []

    filtered_df = df.copy()

    # Text search (basic implementation)
    if q:
        # Search in name, email, lastPosition, resumeText
        pattern = re.compile(q, re.IGNORECASE)

        def text_match(row):
            for field in ['name', 'email', 'lastPosition', 'resumeText']:
                if pd.notna(row[field]) and pattern.search(str(row[field])):
                    return True
            return False

        filtered_df = filtered_df[filtered_df.apply(text_match, axis=1)]

    # Skills filter
    if skills:
        skills_list = [s.strip().lower() for s in skills.split(',')]

        def has_skills(row):
            candidate_skills = set()
            # Add skills from topSkills
            for skill in row['topSkills']:
                candidate_skills.add(skill.lower())
            # Add skills from skills list
            for skill in row['skills']:
                candidate_skills.add(skill['name'].lower())

            return any(skill in candidate_skills for skill in skills_list)

        filtered_df = filtered_df[filtered_df.apply(has_skills, axis=1)]

    # Experience filter
    if experience is not None:
        filtered_df = filtered_df[filtered_df['totalYearsExperience'] >= experience]

    # Location filter
    if location:
        filtered_df = filtered_df[filtered_df['location'].str.contains(location, case=False, na=False)]

    # Application date filter
    if applied_since:
        filtered_df = filtered_df[pd.to_datetime(filtered_df['lastApplicationDate']) >= pd.to_datetime(applied_since)]

    # Sorting and pagination
    filtered_df = filtered_df.sort_values(by="lastApplicationDate", ascending=False)
    total = len(filtered_df)
    start_idx = (page - 1) * limit
    end_idx = min(start_idx + limit, total)

    # Convert to list of dictionaries
    candidates = []
    for idx, row in filtered_df.iloc[start_idx:end_idx].iterrows():
        candidate_dict = row.to_dict()
        candidates.append(CandidateList(**candidate_dict))

    return candidates


@app.post("/api/match-candidates", response_model=List[CandidateList])
async def match_candidates(job_details: JobDetails):
    df = load_candidates_df()

    if df.empty:
        return []

    # Convert DataFrame to list of dictionaries for processing
    candidates = df.to_dict('records')

    # Calculate match scores for each candidate
    matched_candidates = []
    for candidate in candidates:
        match_score = calculate_match_score(candidate, job_details)

        # Only include candidates with a reasonable match score
        if match_score >= 30:  # Threshold can be adjusted
            candidate_copy = dict(candidate)
            candidate_copy["matchScore"] = match_score
            matched_candidates.append(CandidateList(**candidate_copy))

    # Sort by match score
    matched_candidates.sort(key=lambda x: x.matchScore, reverse=True)
    return matched_candidates


# Static files for frontend
# app.mount("/", StaticFiles(directory="frontend/build", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
