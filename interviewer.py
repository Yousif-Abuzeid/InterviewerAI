import os
import json
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import tool
from tavily import TavilyClient
from pydantic import BaseModel, Field
from typing import List, Optional
import nest_asyncio
from crawl4ai import AsyncWebCrawler
import asyncio
import argparse
import shutil

# Environment setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/salma/college/nueral_networks/Dr_mohsen/project/interviewai-459420-ddd596af82a8.json"
os.environ["OPENAI_API_KEY"] = "" # Replace with your OpenAI API key
os.environ["TAVILY_API_KEY"] = ""  # Replace with your Tavily API key
os.environ["GOOGLE_API_KEY"] = "" # Replace with your Google API key
os.environ["GEMINI_API_KEY"] = "" # Replace with your Google API key

# --- Output Directory ---
output_dir = "ai-agent-output"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# --- Async Compatibility ---
nest_asyncio.apply()  # Enable nested async loops for compatibility

# --- Initialize Clients and LLMs ---
search_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
llm = LLM(
    model ='gemini/gemini-2.5-flash-preview-04-17',
    temperature=0,
    api_key=os.environ["GOOGLE_API_KEY"]
)
# llm = LLM(
#     model='gpt-4o-mini',
#     temperature=0,
# )

llm = LLM(
    model ='gemini/gemini-2.5-flash-preview-04-17',
    temperature=0,
    api_key=os.environ["GOOGLE_API_KEY"]
)
llm_gpt = LLM(
    model='gpt-4o-mini',
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"]
)

# --- Data Models ---
class JobAnalysis(BaseModel):
    job_technical_level: str = Field(..., title="Technical level of the job (entry, mid, senior)")
    key_skills: List[str] = Field(..., title="List of key technical skills required for the job", min_items=1)
    include_ps: bool = Field(..., title="Whether to include problem-solving questions in the interview")
    domain_knowledge: List[str] = Field(..., title="List of domain knowledge areas required", min_items=0)

class SearchQueries(BaseModel):
    search_queries: List[str] = Field(..., title="Search queries for interview questions", min_items=1)

class SingleSearchResult(BaseModel):
    title: str = Field(..., title="Title of the search result")
    url: str = Field(..., title="URL of the resource")
    content: str = Field(..., title="Snippet or summary of the resource content")
    score: float = Field(..., title="Relevance score of the result")
    search_query: str = Field(..., title="The query that generated this result")

class AllSearchResults(BaseModel):
    results: List[SingleSearchResult] = Field(..., title="List of search results")

class SkillDetail(BaseModel):
    name: str = Field(..., title="Name of the skill or technology")
    description: Optional[str] = Field(None, title="Brief description of the skill or technology")

class SingleScrapedPage(BaseModel):
    page_url: str = Field(..., title="The URL of the scraped webpage")
    skills: List[SkillDetail] = Field(default_factory=list, title="List of skills extracted from the page")
    technologies: List[SkillDetail] = Field(default_factory=list, title="List of technologies extracted from the page")
    question_examples: List[str] = Field(..., title="List of example interview questions extracted", min_items=1)
    source_type: Optional[str] = Field(None, title="Type of source (e.g., job posting, interview guide, tech blog)")

    @classmethod
    def check_minimum_data(cls, values):
        skills = values.get('skills', [])
        technologies = values.get('technologies', [])
        question_examples = values.get('question_examples', [])
        if not skills and not technologies and not question_examples:
            raise ValueError("At least one of skills, technologies, or question_examples must be non-empty")
        return values

class AllScrapedPages(BaseModel):
    pages: List[SingleScrapedPage] = Field(..., title="List of scraped webpages", min_items=1)

class InterviewQuestion(BaseModel):
    question: str = Field(..., title="Interview question in professional Egyptian Arabic")
    type: str = Field(..., title="Question category")
    difficulty: str = Field(..., title="Difficulty level")

class InterviewScript(BaseModel):
    questions: List[InterviewQuestion] = Field(..., title="List of interview questions", min_items=5, max_items=10)

# --- Tools ---
@tool
def search_engine_tool(query: str):
    """Search for resources related to technical skills and interview questions."""
    try:
        return search_client.search(query)
    except Exception as e:
        return f"Search error: {str(e)}"

async def async_scrape_page(url: str) -> str:
    """Asynchronously scrape a webpage using crawl4ai."""
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url)
            return result.markdown if result and result.markdown else "No content found on page"
    except Exception as e:
        return f"Error scraping page: {str(e)}"

@tool
def web_scraping_tool_for_agent(page_url: str) -> str:
    """Synchronous wrapper for async web scraping."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_scrape_page(page_url))

# --- Agents and Tasks ---
# **Input Processor Agent**
input_processor_agent = Agent(
    role="Input Processor Agent",
    goal="\n".join([
        "To analyze user-provided job position, requirements, and company name.",
        "To identify key technical skills and domain knowledge required for the job.",
        "To determine the technical level of the job (entry, mid, senior).",
        "To determine if problem-solving (PS) questions, such as algorithms or coding challenges, are relevant for the job role."
    ]),
    backstory="\n".join([
        "The agent is the starting point for the TechInterviewerAI workflow, designed to support Arabic-speaking job seekers in the MENA tech market.",
        "It analyzes job details to identify the key technical requirements and skills needed.",
        "It determines the job's technical level and whether problem-solving questions are necessary.",
        "Its analysis ensures the interview aligns with actual job expectations and requirements."
    ]),
    llm=llm,
    verbose=True,
)


input_processor_task = Task(
    description="\n".join([
        "The user provides the following input: job position ({job_position}), requirements ({requirements}), and company name ({company_name}).",
        "Analyze the job details and identify:",
        "1. The technical level of the job (entry, mid, senior) based on the requirements and responsibilities.",
        "2. Extract a list of key technical skills required for this position (e.g., Python, TensorFlow, etc.).",
        "3. Determine if problem-solving (PS) questions (e.g., algorithms, coding challenges) are relevant based on the job role.",
        "4. Identify domain knowledge areas that would be important for this role.",
        "Set include_ps to False only for non-technical roles (e.g., project management) or roles with no coding requirements.",
        "The analysis should consider the context of the MENA tech job market."
    ]),
    expected_output="A JSON object containing job technical level, key skills list, problem-solving relevance, and domain knowledge areas.",
    output_file=os.path.join(output_dir, "step_1_job_analysis.json"),
    output_json=JobAnalysis,
    agent=input_processor_agent
)

# **Search Query Generator Agent**
search_query_generator = Agent(
    role="Search Query Generator",
    goal="Generate tailored search queries for interview questions based on processed input",
    backstory="I am an AI agent specialized in creating effective search queries "
              "to find relevant interview questions based on job details and requirements.",
    verbose=True,
    llm=llm,
)

search_query_generator_task = Task(
    description="\n".join([
        "Using the job position '{job_position}', requirements, and company name '{company_name}':",
        "Read the job analysis from the Input Processor agent.",
        "Generate up to 10 comprehensive search queries for interview questions based on:",
        "1. The job position and company name",
        "2. The key technical skills identified by the Input Processor",
        "3. The technical level of the job",
        "4. Include problem-solving question queries if determined relevant by the Input Processor",
        "5. Domain knowledge areas identified by the Input Processor",
        "Queries must be specific, targeting skills, technologies, or question categories (e.g., 'Python Flask web development interview questions', 'REST API system design interview questions').",
        "Avoid generic queries; focus on job-specific terms (e.g., 'Django backend development' instead of just 'software engineering').",
        "Create a final set of search queries for interview questions."
    ]),
    expected_output="A JSON file containing an array of search queries for interview questions.",
    output_file=os.path.join(output_dir, "step_2_search_queries.json"),
    output_json=SearchQueries,
    agent=search_query_generator
)



# **Research Agent**
research_agent = Agent(
    role="Research Agent",
    goal="\n".join([
        "To retrieve relevant resources based on the search queries provided by the Input Processor Agent.",
        "To collect information on technical skills and interview question types for a specific job role.",
        "To ensure resources are relevant to the MENA tech job market."
    ]),
    backstory="\n".join([
        "The agent is designed to support TechInterviewerAI by gathering high-quality resources for Arabic-speaking job seekers preparing for technical interviews.",
        "It uses search queries to find articles, job descriptions, and interview question examples relevant to the MENA tech industry.",
        "The agent prioritizes reliable and specific content to inform the generation of interview questions."
    ]),
    llm=llm,
    verbose=True,
    tools=[search_engine_tool]
)

research_task = Task(
    description="\n".join([
        "The task is to search for resources based on the English search queries provided in the previous task's output (e.g., 'step_2_search_queries.json').",
        "Collect results for each query, focusing on technical skills, technologies, and interview question types relevant to the job role.",
        "Ignore suspicious links, blogs, or non-relevant content (e.g., unrelated tutorials, general career advice).",
        "Ignore search results with a relevance score less than {score_th} (e.g., 0.7).",
        "Results should prioritize content relevant to the MENA tech job market, such as regional job boards, tech forums, or industry reports.",
        "Return up to 3 results per query to keep the demo manageable.",
        "The search results will be used to generate tailored interview questions in the next stage."
    ]),
    expected_output="A JSON object containing a list of search results, each with a title, URL, content snippet, relevance score, and the query used.",
    output_json=AllSearchResults,
    output_file=os.path.join(output_dir, "step_3_research_results.json"),
    agent=research_agent
)

# **Web Scraper Agent**
web_scraper_agent = Agent(
    role="Web Scraper Agent",
    goal="\n".join([
        "To scrape and analyze webpages from provided URLs to extract skills, technologies, and example interview questions.",
        "To produce structured data relevant to technical interview question generation for the MENA tech market."
    ]),
    backstory="\n".join([
        "The Web Scraper Agent is a specialized component of TechInterviewerAI, designed to collect and analyze web data for Arabic-speaking job seekers.",
        "It processes job postings, tech blogs, and interview guides to extract actionable insights for question generation.",
        "It collaborates with the Question Generator Agent by providing high-quality scraped data."
    ]),
    llm=llm_gpt,
    verbose=True,
    tools=[web_scraping_tool_for_agent]  # Provided tool, not implemented here
)

web_scraper_task = Task(
    description="\n".join([
        "The task is to scrape and analyze webpages using URLs from 'step_3_research_results.json' for the job role of {job_position} with requirements {requirements} and optional company {company_name}.",
        "Use the provided 'web_scraping_tool' to extract data from each URL, including skills (e.g., Python, SQL), technologies (e.g., Django, PostgreSQL), and example interview questions.",
        "For each webpage, produce a structured output with:",
        "  - The page URL.",
        "  - A list of skills with names and optional descriptions.",
        "  - A list of technologies with names and optional descriptions.",
        "  - A list of example interview questions (at least one per page).",
        "  - An optional source type (e.g., job posting, interview guide, tech blog).",
        "Ensure at least one of skills, technologies, or question_examples is non-empty for each page.",
        "Analyze the scraped content to ensure relevance to the MENA tech market and the job role.",
        "The output will be used by the Question Generator Agent to create interview questions."
    ]),
    expected_output="A JSON object containing a list of scraped webpages, each with URL, skills, technologies, question examples, and source type.",
    output_json=AllScrapedPages,
    output_file=os.path.join(output_dir, "step_4_scraped_data.json"),
    agent=web_scraper_agent
)

# **Question Generator Agent**
task_description = (
    "1. Load and analyze scraped job requirements from 'step_4_scraped_data.json' for the target {job_position} and optional {company_name}.\n"
    "2. Extract domain-specific knowledge graph including:\n"
    "   • Primary technology stack components (languages, frameworks, databases, infrastructure)\n"
    "   • Secondary technical requirements and experience thresholds\n"
    "   • Architecture patterns and methodologies mentioned\n"
    "   • Industry-specific tools and platforms\n"
    "3. Generate 6-10 contextually relevant interview questions in Egyptian Arabic with embedded English technical terminology:\n"
    "   • 3-4 Technical depth questions targeting expertise in primary stack components:\n"
    "     - Focus on implementation-level details rather than conceptual definitions\n"
    "     - Reference specific API interfaces, configuration options, or performance characteristics\n"
    "     - Incorporate domain-specific edge cases or optimization techniques\n"
    "   • 2-3 System design or architectural questions that mirror actual production challenges:\n"
    "     - Include concrete constraints (e.g., 'system handles 3000 req/s with p99 latency < 100ms')\n"
    "     - Reference actual services or technologies from the job description\n"
    "     - Require trade-off analysis between competing architectural approaches\n"
    "   • If 'include_ps' flag is True, include 1-2 problem-solving exercises that:\n"
    "     - Are calibrated to senior/staff-level complexity (not basic algorithms)\n"
    "     - Include real-world constraints and edge cases\n"
    "     - Provide specific inputs, expected outputs, and performance requirements\n"
    "4. Calibrate difficulty distribution precisely:\n"
    "   • 30% L3 (mid-level) difficulty: Tests foundational knowledge and basic implementation\n"
    "   • 40% L4 (senior) difficulty: Requires deep technical expertise and experience with edge cases\n"
    "   • 30% L5 (staff) difficulty: Demands systems thinking and advanced optimization knowledge\n"
    "5. Incorporate MENA-specific technology context:\n"
    "   • Reference regional cloud providers or compliance requirements if applicable\n"
    "   • Consider local infrastructure challenges or technology adoption patterns\n"
    "   • Adapt terminology to reflect local tech ecosystem conventions\n"
    "6. Output must strictly conform to the InterviewScript JSON schema with each question object including:\n"
    "   • 'question': The actual interview question in Egyptian Arabic with embedded English terms\n"
    "   • 'type': One of ['technical_depth', 'system_design', 'problem_solving']\n"
    "   • 'difficulty': One of ['L3', 'L4', 'L5']\n"
    "   • 'domain_context': Brief explanation of why this question is relevant to the position\n"
    "   • 'expected_insights': Key technical concepts a strong candidate would address\n"
)

question_generator_agent = Agent(
    role="Enhanced Question Generator",
    goal=(
        "- Craft interview questions deeply rooted in real-world job scenarios and current MENA tech trends.\n"
        "- Leverage scraped data to ensure relevance to {job_position} and {company_name}.\n"
        "- Facilitate seamless Validator review by producing clear, context-rich prompts."
    ),
    backstory=(
        "This agent powers TechInterviewerAI’s core Q&A module, specializing in translating raw job specs into high-fidelity interview prompts."
    ),
    llm=llm,
    verbose=True
)

question_generator_task = Task(
    description=task_description,
    expected_output=(
        "A JSON object matching InterviewScript: an array of 5–10 questions, each with 'question', 'type', and 'difficulty'."
    ),
    output_json=InterviewScript,
    output_file=os.path.join(output_dir, "step_5_interview_script.json"),
    agent=question_generator_agent
)

# --- Interview Crew ---
interview_crew = Crew(
    agents=[
        input_processor_agent,
        search_query_generator,
        research_agent,
        web_scraper_agent,
        question_generator_agent,
    ],
    tasks=[
        input_processor_task,
        search_query_generator_task,
        research_task,
        web_scraper_task,
        question_generator_task,
    ],
    process=Process.sequential
)



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run automated interview based on job description')
    parser.add_argument('--job-description', type=str, help='Job description text', default=None)
    parser.add_argument('--json-file', type=str, help='Path to JSON file containing job description', default=None)
    return parser.parse_args()

def extract_job_details(job_description):
    """Extract relevant details from job description."""
    # Simple extraction - in a real scenario, you might want to use NLP techniques
    lines = job_description.split('\n')
    job_position = lines[0].strip() if lines else "LLM Engineer"
    
    # Extract requirements section
    requirements = ""
    req_start = False
    for line in lines:
        if "Requirements:" in line:
            req_start = True
            continue
        if req_start and line.strip():
            requirements += line.strip() + " "
    
    # Default values if extraction failed
    if not requirements:
        requirements = job_description
    
    # Company name extraction (simplified)
    company_name = "Agentic AI Solutions"
    
    return job_position, requirements, company_name

if __name__ == "__main__":
    args = parse_arguments()
    
    # Load job description from JSON file if path provided
    if not args.job_description and args.json_file:
        try:
            with open(args.json_file, 'r') as f:
                data = json.load(f)
                args.job_description = data.get('jobDescription', '')
        except Exception as e:
            print(f"Error loading JSON file: {e}")
    
    # Or try to load from default location
    if not args.job_description:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            json_file = os.path.join(base_dir, "info.json")
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    args.job_description = data.get('jobDescription', '')
                    print(f"Loaded job description from {json_file}")
        except Exception as e:
            print(f"Error loading default JSON file: {e}")
    
    if not args.job_description:
        # Fall back to example input data if no job description provided
        job_position = "R&D AI ML Developer Intern"
        requirements = """Design and develop scalable AI-driven applications using Python..."""
        company_name = "Siemens"
    else:
        # Extract details from the provided job description
        job_position, requirements, company_name = extract_job_details(args.job_description)
        print(f"Extracted position: {job_position}")
        print(f"Extracted company: {company_name}")
    
    # Example input data
    job_position = "R&D AI ML Developer Intern"
    requirements = """    Design and develop scalable Al-driven applications using Python.
    Build, train, and optimize machine learning models using tools like TensorFlow, PyTorch, Keras, ...etc.
    Process and analyze large datasets to extract valuable insights.
    Design and implement machine learning algorithms tailored to specific business needs.
    Research and implement LLM based techniques (RAG, fine tuning, ...etc.).
    Optimize data flow and processing pipelines to support efficient model inference and training.
    Continuously monitor, troubleshoot, and improve system performance.
    Work closely with engineers, and other stakeholders to ensure Al solutions are aligned with business objectives.
    Document Al models, algorithms, and application workflows.

Qualifications:

    3rd and 4rth year Computer Engineering or Computer Science undergraduates.
    Computer Science fundamentals in object-oriented design, data structures, algorithm design and analysis.
    Strong knowledge of Python and relevant libraries and frameworks commonly used in Al development.
    Extensive experience with machine learning algorithms (supervised, unsupervised, and reinforcement learning).
    Experience with building, maintaining, and deploying production ready Al/LLM based applications.
    Experience in data preprocessing, including cleaning, transformation, and feature engineering to prepare datasets for training
    Good knowledge of (LLM)/GenAl technologies like OpenAl APl, ChatGPT, GPT-4, Bard, Langchain, HuggingFace Transformers, PyTorch and similar.
    Knowledge of Agentic AI approaches.
    Excellent problem-solving and communication skills.
    Ability to work independently and as part of a team.
    Good written and verbal communication skills.""" 
    company_name = "Siemens"
    score_th = 0.7
    print("Starting Interview-eight crew...")
    try:
        result = interview_crew.kickoff(inputs={
            "job_position": job_position,
            "requirements": requirements,
            "company_name": company_name,
            "score_th": score_th
        })
        print("Interviewer crew completed successfully.")
    except Exception as e:
        print(f"Error during crew execution: {e}")
    
    # Format the output script for conductor.py
    # Use the absolute path provided since that's where the file actually exists
    input_script_file = "/home/salma/college/nueral_networks/Dr_mohsen/project/ai-agent-output/step_5_interview_script.json"
    output_script_file = "/home/salma/college/nueral_networks/Dr_mohsen/project/interviewer_script.json"
    try:
        # Directly move the input script file to the output script file
        shutil.copy(input_script_file, output_script_file)
        print(f"Script successfully moved to {output_script_file}")
        

    except Exception as e:
        print(f"Error moving script: {str(e)}")

    # Also verify the file was created
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_script_file)
    if os.path.exists(output_path):
        print(f"Confirmed output file exists at: {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
    else:
        print(f"WARNING: Output file was not created at {output_path}")