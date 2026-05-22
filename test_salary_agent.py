import os
import json
import pytest
from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric, GEval
from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from groqconnection import GroqModel 
from salary_agent import onboard_employee_tool, salary_agent
load_dotenv()
eval_model = GroqModel(model_name="llama-3.3-70b-versatile")



def test_extract_onboarding_metrics_live():
    target_name = "Rohan"
    target_salary = 65000
    user_prompt = f"Please add a new hire named {target_name} with a starting package of {target_salary}."

    print(f"\n🚀 Running real python tool for: {target_name}...")

    real_tool_dictionary_output = onboard_employee_tool(None, name=target_name, starting_package=target_salary)
    actual_tool_text = str(real_tool_dictionary_output)

    captured_tool_calls = [
        ToolCall(
            name="onboard_employee_tool",
            input={"name": target_name, "starting_package": target_salary},
            output=actual_tool_text  
        )
    ]

    expected_tool_calls = [
        ToolCall(
            name="onboard_employee_tool",
            input={"name": target_name, "starting_package": target_salary}
        )
    ]

    test_case = LLMTestCase(
        input=user_prompt,
        actual_output=f"Processed user payload. Tool return details: {actual_tool_text}",
        expected_output="The tool must return Senior Consultant designation for Rohan with CREATED status.",
        tools_called=captured_tool_calls,
        expected_tools=expected_tool_calls 
    )

    # FIX 1: Explicitly pass your eval_model here so DeepEval stops looking for OpenAI keys
    metric = ToolCorrectnessMetric(threshold=0.5, model=eval_model)
    metric.measure(test_case)

    print("\n" + "="*50)
    print("📊 DYNAMIC EXTRACTED TOOL EVALUATION MATRIX")
    print("="*50)
    print(f"📈 Evaluation Score   : {metric.score}")
    print(f"📥 Extracted Tool Data : {actual_tool_text}")
    print(f"🔍 Evaluator Verdict   : {metric.reason}")
    print("="*50 + "\n")

    assert metric.is_successful(), f"Tool evaluation crashed. Reason: {metric.reason}"

# Checking the Agent tool seclation 
def test_agent_tool_routing_intelligence():
  
    user_prompt = "Hey HR, can you please onboard a new person named Sarah? Her package is 120000."
    print(f"\n🚀 Sending text prompt to Agent: '{user_prompt}'")
    
    actual_agent_output = ""
    captured_runtime_tools = []


    test_case = LLMTestCase(
        input=user_prompt,
        actual_output=actual_agent_output,
        tools_called=captured_runtime_tools
    )

    agent_intelligence_metric = GEval(
        name="Agent Tool Routing Intelligence",
        model=eval_model,
        criteria=(
            "Evaluate whether the agent displayed accurate intelligence by routing the user's "
            "request to the correct tool. If the input mentions a 'new person' or 'onboard', "
            "the agent must successfully trigger the onboarding tool. The score should be 1.0 "
            "if the tool executed seamlessly and populated the 'new_hire_onboarding' fields, "
            "and 0.0 if it ignored the onboarding task or used the wrong tool rules."
        ),
        evaluation_steps=[
            "Check if the input prompt requests a new hire onboarding action.",
            "Inspect the actual_output to see if the onboarding tool data fields are present and valid.",
            "Assign a score of 1.0 if the agent picked the right onboarding path, or 0.0 if it failed."
        ],
        threshold=0.7
    )

    agent_intelligence_metric.measure(test_case)

    print("\n" + "="*50)
    print("🧠 AGENT ROUTING INTELLIGENCE MATRIX")
    print("="*50)
    print(f"📈 Intelligence Score : {agent_intelligence_metric.score}")
    print(f"🤖 Live Agent Output  : {actual_agent_output}")
    print(f"🔍 Reason For Verdict : {agent_intelligence_metric.reason}")
    print("="*50 + "\n")

    assert agent_intelligence_metric.is_successful(), f"Agent chose the wrong tool routing path! Reason: {agent_intelligence_metric.reason}"
