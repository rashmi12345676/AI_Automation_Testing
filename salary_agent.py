import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
from groqconnection import GroqModel

# Load your local .env file containing your API key
load_dotenv()
eval_model = GroqModel(model_name="llama-3.3-70b-versatile")

# ==========================================
# 1. DEFINE THE STRUCTURED OUTPUT SCHEMAS
# ==========================================

class PromotionAction(BaseModel):
    employee_name: str = Field(description="Name of the underpaid employee.")
    current_salary: float = Field(description="Their old, low salary.")
    new_recommended_salary: float = Field(description="The new salary calculated inside the good range.")
    promotion_justification: str = Field(description="Reasoning for the automatic promotion.")

class PerformanceTracker(BaseModel):
    file_status: str = Field(description="Status of the tracker file (e.g., CREATED).")
    initial_kpi: str = Field(description="The first Key Performance Indicator assigned to them.")

class OnboardingDetails(BaseModel):
    employee_name: str = Field(description="Name of the newly onboarded employee.")
    assigned_designation: str = Field(description="Designation tier assigned based on package.")
    tracker: PerformanceTracker = Field(description="Performance monitoring registration details.")

class SalaryEvaluationReport(BaseModel):
    average_salary: float = Field(description="The average salary of the entire team.")
    highest_earner: str = Field(description="Name and salary of the top earner.")
    lowest_earner: str = Field(description="Name and salary of the bottom earner.")
    action_taken: str = Field(description="Summary of what the agent did.")
    promotion_details: Optional[PromotionAction] = Field(None, description="Details if an automatic promotion occurred.")
    new_hire_onboarding: Optional[OnboardingDetails] = Field(None, description="Details if a new hire was added and processed.")

# ==========================================
# 2. CREATE THE AGENT WITH BOTH TOOLS
# ==========================================

salary_agent = Agent(
    'groq:llama-3.3-70b-versatile',      
    output_type=SalaryEvaluationReport,  
    instructions=(                       
        "You are an automated HR Compensation and Talent Management Agent. Your job is to analyze team data.\n\n"
        "TASK 1: Always call 'salary_analytics_tool' first to extract raw metrics. "
        "If the lowest paying salary falls below $45,000, automatically trigger a promotion, "
        "raise their salary into the 'good range' ($55,000 - $70,000), and document it.\n\n"
        "TASK 2: Check if there is a 'New Hire' or an employee to add to the system in the input prompt. "
        "If yes, you MUST call 'onboard_employee_tool' to calculate their designation tier, "
        "set up their performance track, and add the configuration to your final report.\n\n"
        "OUTPUT FORMAT RULE: Output your final response as a pure JSON object matching "
        "the schema layout. Do not wrap the JSON in text or fake XML/function tags like '<function=...>'."
    )
)

@salary_agent.tool
def salary_analytics_tool(ctx: RunContext[Any], employee_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyzes an input list of existing employees to calculate the group average, 
    highest earner, and lowest earner.
    """
    print("-> Tool 1: 'salary_analytics_tool' invoked.")
    if not employee_data:
        return {"error": "No employee data provided."}

    # Filter out entries labeled as new hires that are not fully added yet
    valid_emps = [e for e in employee_data if "salary" in e]
    salaries = [emp["salary"] for emp in valid_emps]
    avg_salary = sum(salaries) / len(salaries)
    
    highest_emp = max(valid_emps, key=lambda x: x["salary"])
    lowest_emp = min(valid_emps, key=lambda x: x["salary"])
    
    return {
        "average_salary": round(avg_salary, 2),
        "highest": f"{highest_emp['name']} (${highest_emp['salary']})",
        "lowest_name": lowest_emp['name'],
        "lowest_salary": lowest_emp['salary']
    }

@salary_agent.tool
def onboard_employee_tool(ctx: RunContext[Any], name: str, starting_package: float) -> Dict[str, Any]:
    """
    Onboards a brand new employee to the system. Assigns their designation tier 
    based on starting package and sets up a performance tracker.
    """
    print(f"-> Tool 2: 'onboard_employee_tool' invoked for {name}.")
    
    # Logic to auto-assign designation tier based on package numbers
    if starting_package >= 100000:
        designation = "Lead Specialist"
        kpi = "Lead architecture delivery and mentor junior staff."
    elif starting_package >= 60000:
        designation = "Senior Consultant"
        kpi = "Deliver modular project assets independently."
    else:
        designation = "Associate Professional"
        kpi = "Complete core module onboarding tasks and track learning hours."

    return {
        "employee_name": name,
        "assigned_designation": designation,
        "performance_track_status": "CREATED",
        "initial_kpi": kpi
    }

# ==========================================
# 3. EXECUTE THE AGENT WITH COMPLEX TEST DATA
# ==========================================

# Current payroll roster containing an underpaid employee ("Bob") 
# and a notice to add a completely new employee ("Evan")
complex_hr_input = {
    "current_roster": [
        {"name": "Alice", "role": "Senior Engineer", "salary": 110000},
        {"name": "Bob", "role": "Junior Associate", "salary": 38000},  # Triggers promotion
        {"name": "Charlie", "role": "Manager", "salary": 85000}
    ],
    "new_hire_to_add": {
        "name": "Evan",
        "starting_package": 105000  # Triggers tool onboarding
    }
}

def run_hr_evaluation():
    print("Running multi-tool automated HR Agent evaluation...\n")
    prompt = f"Process this HR data payload: {complex_hr_input}"
    
    result = salary_agent.run_sync(prompt)
    report: SalaryEvaluationReport = result.output 
    
    print("\n=== AGENT EVALUATION REPORT ===")
    print(f"Team Average Salary : ${report.average_salary:,}")
    print(f"Highest Paid        : {report.highest_earner}")
    print(f"Lowest Paid         : {report.lowest_earner}")
    print(f"Action Summary      : {report.action_taken}")
    
    if report.promotion_details:
        print("\n--- AUTOMATIC PROMOTION RECORD ---")
        print(f"Employee Name       : {report.promotion_details.employee_name}")
        print(f"Old Base Salary     : ${report.promotion_details.current_salary:,}")
        print(f"New Base Salary     : ${report.promotion_details.new_recommended_salary:,}")
        print(f"HR Justification    : {report.promotion_details.promotion_justification}")

    if report.new_hire_onboarding:
        print("\n--- AUTOMATIC NEW HIRE ONBOARDING RECORD ---")
        print(f"Employee Name       : {report.new_hire_onboarding.employee_name}")
        print(f"Designation Tier    : {report.new_hire_onboarding.assigned_designation}")
        print(f"Performance Track   : {report.new_hire_onboarding.tracker.file_status}")
        print(f"Assigned KPI Target : {report.new_hire_onboarding.tracker.initial_kpi}")

if __name__ == "__main__":
    run_hr_evaluation()
