from dotenv import load_dotenv
import streamlit as st
import asyncio
import os
import json
from groq import AsyncGroq  # ⚡ Switched to official Groq SDK
from playwright.async_api import async_playwright

# Load environment variables from .env
load_dotenv()

# Setup page config
st.set_page_config(page_title="AI QA Testing Agent", page_icon="🤖", layout="wide")

# Initialize Groq Client
groq_key = os.environ.get("GROQ_API_KEY", "")
if not groq_key:
    st.error("🔑 Please export your GROQ_API_KEY in your environment or .env file to run this application.")
client = AsyncGroq(api_key=groq_key)

# Define Tools for Browser Automation
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "Navigate the browser to a specific URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The exact absolute web URL to load"}
                },
                "required": ["url"],
                "additionalProperties": False 
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_element",
            "description": "Click an element on the page using a CSS selector or text format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "e.g., 'button', '#submit', 'text=Log In'"}
                },
                "required": ["selector"],
                "additionalProperties": False
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into an input form element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "e.g., 'input[type=email]', '#username'"},
                    "text": {"type": "string", "description": "The string to type"}
                },
                "required": ["selector", "text"],
                "additionalProperties": False
            },
        },
    }
]

# Core Async Testing Engine
async def run_ai_test(target_url, criteria, status_box, logs_container):
    async with async_playwright() as p:
        # Launch browser headlessly for background processing
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI QA Engineer. Your job is to test the application based on the user's criteria.\n\n"
                    "CRITICAL FOR TOOL USAGE:\n"
                    "1. When calling a tool, you MUST use the strict built-in tool call JSON format. Never output XML/HTML tags like '<function=...>' or raw text descriptions of your actions.\n"
                    "2. You must evaluate the webpage structure, call appropriate tools, and report progress.\n"
                    "3. If the application passes the criteria, reply exactly with: 'TEST RESULT: PASSED -> [detailed reason]'.\n"
                    "4. If you find a bug or criteria fails, reply exactly with: 'TEST RESULT: FAILED -> [detailed reason]'."
                )
            },
            {
                "role": "user",
                "content": f"Target App URL: {target_url}\n\nAcceptance Criteria to verify:\n{criteria}"
            }
        ]

        logs = []
        
        for turn in range(12):
            # Dynamic tool enforcement logic for Groq
            current_tool_choice = "required" if turn == 0 else "auto"
            
            try:
                response = await client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice=current_tool_choice,
                    temperature=0.0  # Force deterministic tool selection matching schemas perfectly
                )
            except Exception as e:
                error_msg = f"❌ **Groq API Error:** {str(e)}"
                logs.append(error_msg)
                with logs_container:
                    st.error(error_msg)
                break
            
            # Grabbing the first choice message safely
            response_message = response.choices[0].message
            messages.append(response_message)

            # Check for thinking outputs or final declarations
            if response_message.content:
                thought = response_message.content
                if "TEST RESULT:" in thought:
                    await browser.close()
                    return thought, logs
                else:
                    logs.append(f"🧠 **Thinking:** {thought}")
                    with logs_container:
                        st.markdown(f"🧠 **Thinking:** {thought}")

            # Execute tool actions if prompted by LLM
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    
                    action_log = f"⚡ **Action:** Executing `{func_name}` with `{args}`"
                    logs.append(action_log)
                    with logs_container:
                        st.markdown(action_log)
                    
                    tool_result = ""
                    try:
                        if func_name == "navigate_to":
                            await page.goto(args["url"])
                            tool_result = f"Successfully loaded page: {args['url']}"
                        elif func_name == "click_element":
                            await page.click(args["selector"], timeout=5000)
                            tool_result = f"Successfully clicked element: '{args['selector']}'"
                        elif func_name == "type_text":
                            await page.fill(args["selector"], args["text"], timeout=5000)
                            tool_result = f"Successfully typed text into field: '{args['selector']}'"
                    except Exception as e:
                        tool_result = f"Execution failed. Error details: {str(e)}"
                    
                    state_log = f"📊 **Browser Output:** {tool_result}"
                    logs.append(state_log)
                    with logs_container:
                        st.markdown(state_log)
                    
                    # Capture page context for the agent
                    try:
                        await page.wait_for_load_state("networkidle", timeout=1500)
                    except:
                        pass
                    
                    page_text = await page.evaluate("() => document.body.innerText")
                    
                    tool_output_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": json.dumps({
                            "action_status": tool_result,
                            "current_page_text_snapshot": page_text[:2000]
                        })
                    }
                    messages.append(tool_output_message)
                    
        await browser.close()
        return "TEST RESULT: TIMEOUT -> Maximum test execution steps reached.", logs

# --- Build the UI Elements ---
st.title("🤖 Autonomous AI QA Testing Bot")
st.markdown("Provide your feature URL along with your acceptance criteria below. The AI will navigate the page and monitor results live.")

# Layout Columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Test Requirements Configuration")
    url_input = st.text_input("Application Target URL", placeholder="https://example.com")
    criteria_input = st.text_area("Acceptance Criteria (Plain English or Gherkin)", 
                                  placeholder="1. Navigate to page\n2. Confirm certain text content is present.",
                                  height=200)
    
    run_btn = st.button("🚀 Execute Autonomous Test", use_container_width=True)

with col2:
    st.subheader("📊 Live Testing Execution & Monitoring Results")
    
    if run_btn:
        if not url_input or not criteria_input:
            st.warning("⚠️ Please provide both a valid application URL and Acceptance Criteria details.")
        else:
            status_box = st.info("⚙️ Initializing autonomous browser session... please wait.")
            logs_container = st.container(border=True)
            
            # Execute async framework loop inside Streamlit context
            with st.spinner("AI Agent is working on the application..."):
                final_outcome, test_logs = asyncio.run(run_ai_test(url_input, criteria_input, status_box, logs_container))
            
            # Display finalized evaluations
            status_box.empty()
            if "PASSED" in final_outcome:
                st.success(f"🏁 {final_outcome}")
            else:
                st.error(f"🏁 {final_outcome}")
    else:
        st.info("💡 Fill out the parameters on the left and click execute to see live step logs.")
