#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -qU crewai[tools]==0.95.0')


# In[ ]:


get_ipython().system('pip install playwright')
get_ipython().system('playwright install')
get_ipython().system('pip install nest_asyncio')

get_ipython().system('playwright install --with-deps')



# In[ ]:


get_ipython().system('pip install crawl4ai')


# In[ ]:


from crewai import Agent, Task, Crew, Process, LLM
# import agentops
from crewai.tools import tool
import os
from pydantic import BaseModel, Field
from typing import List
get_ipython().system('pip install langchain_openai')
get_ipython().system('pip install tavily-python')
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from google.colab import userdata


# In[ ]:


os.environ["OPENAI_API_KEY"]=userdata.get('openai-colab')
# os.environ["AGENTOPS_ENABLED"] = "false"
# os.environ["AGENTOPS_API_KEY"]=""

# agentops.init(
#     api_key=userdata.get('agentops-colab'),
#     skip_auto_end_session=True
# )




llm = LLM(model="gpt-4o-mini",temperature=0)
llm_advanced = LLM(model="gpt-4o",temperature=0)

# In[ ]:


output_dir = "./ai-agent-output"
os.makedirs(output_dir, exist_ok=True)

# basic_llm = LLM(model="gpt-4o",temperature=0)
search_client = TavilyClient(
    api_key="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"

)


# ## Agent A 
# ### Agent A is the input processor Agent and will be responsible for:
# - Receiving the input from the user
# - Processing the input
# - Sending the processed input to Agent B
# 

# In[ ]:


class JobAnalysis(BaseModel):
    job_technical_level: str = Field(..., title="Technical level of the job (entry, mid, senior)")
    key_skills: List[str] = Field(..., title="List of key technical skills required for the job", min_items=1)
    include_ps: bool = Field(..., title="Whether to include problem-solving questions in the interview")
    domain_knowledge: List[str] = Field(..., title="List of domain knowledge areas required", min_items=0)


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


# ## Agent B
# ### Agent B get the processed input from Agent A and will be responsible for:
# - Generating search queries based on the processed input

# In[ ]:


class SearchQueries(BaseModel):
    search_queries: List[str] = Field(..., title="Search queries for interview questions", min_items=1)

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




# ## Agent C
# ### Agent C get the search queries from Agent B and will be responsible for:
# - Searching the web for relevant information
# - Collecting the information
# - Sending the information to Agent D

# In[ ]:


class SingleSearchResult(BaseModel):
    title: str = Field(..., title="Title of the search result")
    url: str = Field(..., title="URL of the resource")
    content: str = Field(..., title="Snippet or summary of the resource content")
    score: float = Field(..., title="Relevance score of the result")
    search_query: str = Field(..., title="The query that generated this result")

# Define the output model for all search results
class AllSearchResults(BaseModel):
    results: List[SingleSearchResult] = Field(..., title="List of search results")

# Mock or actual search tool (replace with your search client, e.g., Google Custom Search API)
@tool
def search_engine_tool(query: str):
    """Useful for searching resources related to technical skills and interview questions. Use this to find current information about job-specific skills and question types."""
    return search_client.search(query)  # Replace with actual search implementation


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


# ## Agent D 
# ### Agent D get the URLs from Agent C and will be responsible for:
# - Extracting the relevant information from the URLs
# - Sending the extracted information to Agent E

# In[ ]:


# --- Data Models ---
from pydantic import BaseModel, Field
from typing import List, Optional

import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()


class SkillDetail(BaseModel):
    name: str = Field(..., title="Name of the skill or technology (e.g., Python, Flask)")
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



from crawl4ai import AsyncWebCrawler
import asyncio

async def async_scrape_page(url: str) -> str:
    """
    Asynchronously scrape a webpage using crawl4ai AsyncWebCrawler and return its content.

    Args:
        url: The URL of the webpage to scrape

    Returns:
        str: The scraped content of the webpage in markdown format, or an error message if scraping fails
    """
    try:
        print(f"Starting to scrape: {url}")

        # Create the AsyncWebCrawler and scrape the page
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url)

            if not result or not result.markdown:
                print("No content found when crawling the page")
                return "No content found on page"

            # Return the markdown content
            content = result.markdown
            print(f"Successfully crawled page with {len(content)} characters of content")
            return content

    except Exception as e:
        import traceback
        print(f"Error scraping page: {str(e)}")
        print(traceback.format_exc())
        return f"Error scraping page: {str(e)}"

# --- Synchronous wrapper for the async function ---
@tool
def web_scraping_tool_for_agent(page_url: str) -> str:
    """
    A synchronous wrapper for the async scraping function to use with CrewAI.

    Args:
        page_url: The URL of the webpage to scrape

    Returns:
        str: The scraped content of the webpage in markdown format
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_scrape_page(page_url))

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
    llm=llm,
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



# In[ ]:





# ## Agent E
# ### Agent E get the extracted information from Agent D and will be responsible for:
# - Generating a number of interview questions based on the extracted information

# In[ ]:


from typing import Optional, List
import os
import json
from pydantic import BaseModel, Field

# --- Agent 3: Enhanced Interview Question Generator (With Web Scraping Integration) ---
class InterviewQuestion(BaseModel):
    question: str = Field(
        ...,
        title="Interview question in professional Egyptian Arabic",
        description=(
            "A realistic, job-focused question phrased in formal Egyptian Arabic with English technical terms. "
            "Include specific scenarios, example inputs/outputs, or debugging contexts."
        )
    )
    type: str = Field(
        ...,
        title="Question category",
        description="One of: 'technical' for direct skill checks, 'problem-solving' for algorithmic puzzles, 'scenario-based' for real-world case studies."
    )
    difficulty: str = Field(
        ...,
        title="Difficulty level",
        description="One of: 'easy' (basic knowledge), 'medium' (intermediate complexity), 'hard' (advanced or multi-step problem)."
    )

class InterviewScript(BaseModel):
    """
    Schema for the final script of interview questions.
    Contains between 5 and 10 entries, ensuring diverse coverage of technical skills, algorithms, and real-world scenarios.
    """
    questions: List[InterviewQuestion] = Field(
        ...,
        title="List of interview questions",
        min_items=5,
        max_items=10,
        description=(
            "Aggregate of 3–5 technical questions, 1–2 algorithmic problem-solving tasks (if include_ps=True), "
            "and 1–2 scenario-based questions simulating on-the-job challenges."
        )
    )

# Instantiate the enhanced agent for question generation
task_description = (
    "1. Load scraped job requirements from 'step_4_scraped_data.json' for the target {job_position}, including optional {company_name}.\n"
    "2. Examine key skills, tools, and responsibilities to inform question content.\n"
    "3. Generate 5–10 interview questions in Egyptian Arabic with embedded English tech terms: \n"
    "   • At least 3 core technical questions on primary stack (e.g., APIs, databases, frameworks).\n"
    "   • If 'include_ps' flag is True, include 1–2 LeetCode-style coding problems with sample inputs/outputs.\n"
    "   • Up to 2 scenario-based questions mirroring real tasks (e.g., optimizing MongoDB queries under high read load, diagnosing a Kubernetes pod crash).\n"
    "4. Balance difficulty: ~40% easy, ~40% medium, ~20% hard.\n"
    "5. Embed concrete details: service names, error messages, performance metrics, expected deliverables.\n"
    "6. Output must strictly conform to the InterviewScript JSON schema."
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
    llm=llm_advanced,
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



# In[ ]:





# In[ ]:





# In[ ]:





# ## Validation Agent
# ### Not used now

# In[ ]:


# class ValidatedQuestion(BaseModel):
#     question: str = Field(..., title="The validated or rewritten interview question in professional Egyptian Arabic with English technical terms")
#     type: str = Field(..., title="Type of question (technical or problem-solving)")
#     difficulty: str = Field(..., title="Difficulty level (easy, medium, hard)")
#     validation_status: str = Field(..., title="Status (valid, rewritten)")
#     validation_note: str = Field(..., title="Explanation of validation or feedback provided")

# class ValidatedScript(BaseModel):
#     questions: List[ValidatedQuestion] = Field(
#         ...,
#         title="List of validated interview questions",
#         min_items=5,
#         max_items=10
#     )

# question_validator_agent = Agent(
#     role="Question Validator Agent",
#     goal="\n".join([
#         "To review and validate interview questions, ensuring they meet quality standards for a technical interview in the MENA tech market.",
#         "To provide feedback to the Question Generator Agent to rewrite invalid questions, maintaining professional Egyptian Arabic, English technical terms, and LeetCode-style problem-solving questions.",
#         "To produce a final validated script with clear, specific, and job-relevant questions."
#     ]),
#     backstory="\n".join([
#         "The agent is an essential part of TechInterviewerAI, ensuring top-quality technical interview questions for Arabic-speaking job seekers in the MENA region.",
#         "It reviews questions for clarity, specificity, and alignment with job requirements, using professional Egyptian Arabic and English technical terms.",
#         "It collaborates with the Question Generator Agent, providing targeted feedback to refine questions and simulate real technical interviews."
#     ]),
#     llm=llm,
#     verbose=True,
# )

# question_validator_task = Task(
#     description="\n".join([
#         "The task is to validate the interview questions in 'step_4_interview_script.json' for the job role of {job_position} with requirements {requirements} and optional company {company_name}.",
#         "Use the 'include_ps' flag from 'step_1_input_processor_queries.json' to verify if LeetCode-style problem-solving (PS) questions are required.",
#         "Validate each question based on the following criteria:",
#         "  - Language: Professional Egyptian Arabic with English technical terms (e.g., 'REST API', 'Python').",
#         "  - Type: Only technical or problem-solving questions (no behavioral). PS questions must be LeetCode-style, requiring coded solutions with detailed descriptions (e.g., problem statement, input/output, constraints, edge cases).",
#         "  - Difficulty: Mix of ~40% easy (e.g., input validation), ~40% medium (e.g., array manipulation), ~20% hard (e.g., complex data structures).",
#         "  - Clarity: Clear, concise, and unambiguous, with descriptive context for technical questions and detailed problem statements for PS questions.",
#         "  - Specificity: Relevant to the job’s skills/technologies (e.g., Python, REST APIs, SQL for a Software Engineer).",
#         "  - MENA Relevance: Aligned with MENA tech trends (e.g., cloud computing, mobile development).",
#         "  - Cultural Appropriateness: Professional tone, avoiding slang or sensitive topics.",
#         "For each question:",
#         "  - If valid, mark as 'valid' with a note explaining why.",
#         "  - If invalid (e.g., vague, wrong type, insufficient detail, incorrect difficulty), provide feedback (e.g., 'Problem-solving question lacks input/output examples and constraints') and mark for rewriting.",
#         "Trigger the Question Generator Agent to rewrite invalid questions by passing feedback, ensuring the revised questions meet all criteria.",
#         "Combine valid and rewritten questions into a final script with 5–10 questions, maintaining the difficulty mix and job relevance.",
#         "The validated script will be used for a mock technical interview."
#     ]),
#     expected_output="A JSON object containing a list of 5–10 validated or rewritten interview questions in professional Egyptian Arabic with English technical terms, each with a question text, type (technical or problem-solving), difficulty (easy, medium, hard), validation status (valid, rewritten), and validation note.",
#     output_json=ValidatedScript,
#     output_file=os.path.join(output_dir, "step_5_validated_questions.json"),
#     agent=question_validator_agent
# )


# In[ ]:


# class FinalInterviewScript(BaseModel):
#     questions: List[InterviewQuestion] = Field(
#         ..., title="Final list of interview questions", min_items=5, max_items=10
#     )


# question_revision_task = Task(
#     description="\n".join([
#         "Using the initial questions from 'step_4_interview_script.json' and validation results",
#         "from 'step_5_validated_questions.json', produce the final interview script by:",
#         "- Keeping questions marked as 'valid' as is.",
#         "- Revising questions marked as 'invalid' based on the feedback in the validation notes.",
#         "Ensure revised questions address the specific issues mentioned, remain in professional",
#         "Egyptian Arabic with English technical terms, and maintain the required format.",
#         "The final script should have 5-10 questions with a mix of types and difficulties."
#     ]),
#     expected_output="A JSON object with the final list of 5-10 interview questions.",
#     output_json=FinalInterviewScript,
#     output_file=os.path.join(output_dir, "step_6_final_interview_script.json"),
#     agent=question_generator_agent
# )


# In[ ]:





# ## Interview Crew

# In[ ]:


interviewCrew= Crew(
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
    proccess=Process.sequential
)


# In[ ]:





# In[ ]:


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

result = interviewCrew.kickoff(inputs={
    "job_position": job_position,
    "requirements": requirements,
    "company_name": company_name,
    "score_th": score_th
})


# In[ ]:




