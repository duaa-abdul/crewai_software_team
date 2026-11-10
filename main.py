
import os
from dotenv import load_dotenv

# =====================================
# FIX GROQ CACHE BREAKPOINT ISSUE
# =====================================

import crewai.llms.cache as crewai_cache

crewai_cache.mark_cache_breakpoint = lambda msg: msg

from crewai import Agent, Task, Crew, Process, LLM

# =====================================
# LOAD ENVIRONMENT VARIABLES
# =====================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Please add GROQ_API_KEY to your .env file")

# Create output folder
os.makedirs("outputs", exist_ok=True)

# =====================================
# GROQ LLM CONFIGURATION
# =====================================

llm = LLM(
    model="groq/openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0.2,
    max_tokens=1800
)

# =====================================
# AGENT 1: PROJECT PLANNER
# =====================================

project_planner = Agent(
    role="Project Planner",
    goal="Analyze software requirements and create an implementation plan.",
    backstory=(
        "You are an experienced Software Architect. "
        "You understand software requirements, identify features, "
        "define Python functions and prepare development plans."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# =====================================
# AGENT 2: PYTHON DEVELOPER
# =====================================

python_developer = Agent(
    role="Python Developer",
    goal="Generate modular and error-handled Python code.",
    backstory=(
        "You are an expert Python Backend Developer. "
        "You write clean, modular and executable Python code. "
        "You use functions, meaningful variables and proper error handling."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# =====================================
# AGENT 3: QA ENGINEER
# =====================================

qa_engineer = Agent(
    role="QA Engineer",
    goal="Generate unit tests and edge cases for Python code.",
    backstory=(
        "You are a professional Software Tester. "
        "You create unit tests, test normal inputs, invalid inputs "
        "and edge cases using Python unittest."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# =====================================
# AGENT 4: CODE REVIEWER
# =====================================

code_reviewer = Agent(
    role="Code Reviewer",
    goal="Review Python code and prepare the final development report.",
    backstory=(
        "You are a Senior Python Developer. "
        "You identify bugs, security issues, code quality problems "
        "and recommend improvements."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# =====================================
# TASK 1: REQUIREMENT ANALYSIS
# =====================================

planning_task = Task(
    description="""
    Analyze this software requirement:

    {requirement}

    Prepare an implementation plan containing:
    1. Project overview
    2. Main features
    3. Required Python functions
    4. Inputs and outputs
    5. Error handling
    6. Development steps

    Keep the plan clear and practical.
    """,
    expected_output="A complete software implementation plan.",
    agent=project_planner,
    output_file="outputs/implementation_plan.md"
)

# =====================================
# TASK 2: PYTHON CODE GENERATION
# =====================================

development_task = Task(
    description="""
    Develop a complete Python application for:

    {requirement}

    Follow the implementation plan from the previous task.

    Requirements:
    1. Write modular Python code.
    2. Use separate functions.
    3. Add error handling.
    4. Use meaningful variable names.
    5. Follow Python coding standards.
    6. Make the code executable.

    Return only valid Python source code.
    Do not use Markdown code fences.
    """,
    expected_output="Complete executable Python source code.",
    agent=python_developer,
    context=[planning_task],
    output_file="outputs/generated_code.py"
)

# =====================================
# TASK 3: QA TEST CASE GENERATION
# =====================================

testing_task = Task(
    description="""
    Create unit tests for the generated Python application.

    Original requirement:
    {requirement}

    Analyze the developer's code.

    Include:
    1. Normal test cases
    2. Invalid input tests
    3. Edge cases
    4. Expected results
    5. Error handling tests

    Use Python unittest framework.
    Import functions from generated_code.py.

    Return only valid Python test code.
    Do not use Markdown code fences.
    """,
    expected_output="Complete Python unittest test file.",
    agent=qa_engineer,
    context=[planning_task, development_task],
    output_file="outputs/test_cases.py"
)

# =====================================
# TASK 4: CODE REVIEW
# =====================================

review_task = Task(
    description="""
    Review the generated Python code and test cases.

    Original requirement:
    {requirement}

    Check:
    1. Code quality
    2. Possible bugs
    3. Error handling
    4. Security issues
    5. Missing functionality
    6. Test coverage
    7. Recommended improvements

    Clearly explain all findings.

    Do not claim that tests were executed.
    Only report actual execution results if available.
    """,
    expected_output="Detailed code review with bugs and improvements.",
    agent=code_reviewer,
    context=[planning_task, development_task, testing_task],
    output_file="outputs/code_review.md"
)

# =====================================
# TASK 5: FINAL DEVELOPMENT REPORT
# =====================================

report_task = Task(
    description="""
    Prepare the final software development report.

    Original requirement:
    {requirement}

    Use the outputs from all previous tasks.

    Include:
    1. Project summary
    2. Implementation approach
    3. Generated functions
    4. Test cases summary
    5. Code review findings
    6. Improvements
    7. Final development status

    Clearly distinguish generated tests from executed tests.
    """,
    expected_output="Complete final software development report.",
    agent=code_reviewer,
    context=[
        planning_task,
        development_task,
        testing_task,
        review_task
    ],
    output_file="outputs/final_report.md"
)

# =====================================
# CREATE SOFTWARE DEVELOPMENT CREW
# =====================================

software_crew = Crew(
    agents=[
        project_planner,
        python_developer,
        qa_engineer,
        code_reviewer
    ],
    tasks=[
        planning_task,
        development_task,
        testing_task,
        review_task,
        report_task
    ],
    process=Process.sequential,
    verbose=True
)

# =====================================
# RUN APPLICATION
# =====================================

if __name__ == "__main__":

    print("\n====================================")
    print("   AI SOFTWARE DEVELOPMENT TEAM")
    print("====================================")

    requirement = input(
        "\nEnter your software requirement: "
    ).strip()

    if not requirement:
        print("Please enter a valid requirement.")

    else:
        try:
            print("\nAI Agents are starting their work...\n")

            result = software_crew.kickoff(
                inputs={
                    "requirement": requirement
                }
            )

            print("\n====================================")
            print("   FINAL DEVELOPMENT RESULT")
            print("====================================\n")

            print(result)

            print("\n====================================")
            print("All tasks completed!")
            print("Check your outputs folder.")
            print("====================================")

        except Exception as e:
            print("\nCrewAI Execution Error:")
            print(str(e))
