import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI



# -----------------------------
# Setup
# -----------------------------
load_dotenv()

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to your .env file locally or Streamlit secrets when deployed.")
    st.stop()

client = OpenAI(api_key=api_key)

MODEL_NAME = "gpt-5.5"

FEEDBACK_FORM_URL = "https://forms.gle/KV3b9NiTWuHetsHf8"

SYSTEM_INSTRUCTION = """
You are an accountability and goal-setting assistant.

Your job is to help users turn vague goals into clear, SMART, trackable action systems.

You are inspired by:
- Brian Tracy’s goal-setting principles: clear written goals, deadlines, priority, discipline, and action.
- James Clear’s habit/system principles: small repeatable actions, systems over motivation, tracking, environment design, and consistency.

Do not copy long passages from any author. Use short references or paraphrased principles only.

Main flow:
1. Receive the user’s vague goal.
2. Help convert it into a SMART goal:
   - Specific
   - Measurable
   - Achievable
   - Relevant
   - Time-bound
3. For each SMART part, explain why it matters in one short sentence.
4. Suggest a close example the user may accept or edit.
5. Once SMART details are complete, break the goal into manageable steps.
6. Assign each step to a realistic time block.
7. Make sure the final step deadline matches the user’s Time-bound deadline.
8. Return the plan as structured JSON that can be used to build a tracking table.
9. Ask the user whether they are satisfied with the generated plan.
10. If they are not satisfied, ask clarification questions about:
    - missing details
    - obstacles
    - available time
    - difficulty level
    - preferred schedule
11. Regenerate the plan using the full conversation history.
12. Continue this loop until the user is satisfied.

Tone:
Firm, practical, encouraging, clear, and not too soft.

Output requirements:
Return valid JSON only when asked to generate or regenerate the plan.
Do not include markdown in JSON responses.
"""


SMART_FIELDS = {
    "specific": {
        "label": "S — Specific",
        "reason": "A clear goal gives your mind one exact target to work toward.",
        "principle": "Brian Tracy-style principle: vague goals create vague action; clear goals create focused action.",
    },
    "measurable": {
        "label": "M — Measurable",
        "reason": "You need a way to know if you are actually making progress.",
        "principle": "James Clear-style principle: tracking makes progress visible and keeps the system honest.",
    },
    "achievable": {
        "label": "A — Achievable",
        "reason": "The goal should stretch you, but it still has to match your current reality.",
        "principle": "Brian Tracy-style principle: goals should challenge you while still being believable enough to act on.",
    },
    "relevant": {
        "label": "R — Relevant",
        "reason": "The goal must connect to something that actually matters to your life.",
        "principle": "James Clear-style principle: habits stick better when they connect to the identity you want to build.",
    },
    "time_bound": {
        "label": "T — Time-bound",
        "reason": "A deadline creates urgency and helps the plan become real.",
        "principle": "Brian Tracy-style principle: goals need deadlines because deadlines force priority and action.",
    },
}


# -----------------------------
# Session State
# -----------------------------
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "goal" not in st.session_state:
        st.session_state.goal = ""

    if "smart_inputs" not in st.session_state:
        st.session_state.smart_inputs = {
            "specific": "",
            "measurable": "",
            "achievable": "",
            "relevant": "",
            "time_bound": "",
        }

    if "smart_suggestions" not in st.session_state:
        st.session_state.smart_suggestions = {}

    if "plan" not in st.session_state:
        st.session_state.plan = None

    if "satisfied" not in st.session_state:
        st.session_state.satisfied = False

    if "clarification_notes" not in st.session_state:
        st.session_state.clarification_notes = ""

    if "page" not in st.session_state:
        st.session_state.page = "goal_setup"


# -----------------------------
# OpenAI Helpers
# -----------------------------
def call_model(user_prompt: str) -> str:
    """
    Sends full conversation history plus the newest user prompt.
    This gives each API call memory of the previous app conversation.
    """

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_prompt
    })

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=SYSTEM_INSTRUCTION,
        input=st.session_state.chat_history,
    )

    assistant_text = response.output_text

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_text
    })

    return assistant_text


def extract_json(text: str) -> Dict[str, Any]:
    """
    Attempts to safely extract JSON from the model response.
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", text, re.DOTALL)

    if not json_match:
        raise ValueError("No JSON object found in model response.")

    return json.loads(json_match.group(0))


# -----------------------------
# Goal Suggestion Logic
# -----------------------------
def generate_smart_suggestions(goal: str) -> Dict[str, str]:
    prompt = f"""
The user gave this goal:

{goal}

Generate close SMART suggestions for this goal.

Return valid JSON only in this exact format:

{{
  "specific": "suggested specific version",
  "measurable": "suggested measurable version",
  "achievable": "suggested achievable version",
  "relevant": "suggested relevant version",
  "time_bound": "suggested time-bound version"
}}
"""

    response_text = call_model(prompt)
    return extract_json(response_text)


def generate_plan(goal: str, smart_inputs: Dict[str, str], clarification_notes: str = "") -> Dict[str, Any]:
    prompt = f"""
Create a full accountability plan from this goal and SMART information.

Original goal:
{goal}

SMART details:
Specific: {smart_inputs["specific"]}
Measurable: {smart_inputs["measurable"]}
Achievable: {smart_inputs["achievable"]}
Relevant: {smart_inputs["relevant"]}
Time-bound: {smart_inputs["time_bound"]}

Extra clarification or user feedback:
{clarification_notes}

Rules:
- Break the goal into manageable steps from start to completion.
- Assign each step to a time block.
- The final step deadline must match the Time-bound deadline.
- Include likely obstacles.
- Include one practical accountability check for each step.
- Return valid JSON only.
- Do not use markdown.

Return JSON in this exact structure:

{{
  "goal_summary": "clear rewritten goal",
  "smart_goal": {{
    "specific": "...",
    "measurable": "...",
    "achievable": "...",
    "relevant": "...",
    "time_bound": "..."
  }},
  "overall_deadline": "...",
  "likely_obstacle": "...",
  "steps": [
    {{
      "step_number": 1,
      "step_name": "...",
      "description": "...",
      "time_block": "...",
      "deadline": "...",
      "accountability_check": "...",
      "status": "Not Started",
      "notes": ""
    }}
  ],
  "today_next_action": "one realistic action the user can do today"
}}
"""

    response_text = call_model(prompt)
    return extract_json(response_text)


def plan_to_dataframe(plan: Dict[str, Any]) -> pd.DataFrame:
    steps = plan.get("steps", [])
    return pd.DataFrame(steps)

def show_feedback_link(position="top"):
    if position == "top":
        st.markdown(
            f"""
            <div style="text-align: right; margin-bottom: 15px;">
                <a href="{FEEDBACK_FORM_URL}" target="_blank">
                    📝 Give Feedback
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown("---")
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px; margin-bottom: 20px;">
                <p>Help improve this app.</p>
                <a href="{FEEDBACK_FORM_URL}" target="_blank">
                    📝 Give Feedback
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    
def show_daily_checkin_page():

    show_feedback_link("top")
    
    st.title("📅 Daily Check-In Page")
    st.info("You are now on the Daily Check-In page. Update your progress below.")

    if not st.session_state.plan:
        st.warning("No plan found yet. Please create a goal plan first.")

        if st.button("Go back to goal setup"):
            st.session_state.page = "goal_setup"
            st.rerun()

        return

    plan = st.session_state.plan
    steps = plan.get("steps", [])

    st.subheader("Your Goal")
    st.write(plan.get("goal_summary", ""))

    st.subheader("Today's Focus")
    st.success(plan.get("today_next_action", "No next action found."))

    st.subheader("Check In on Your Steps")

    updated_steps = []

    for index, step in enumerate(steps):
        with st.container():
            st.markdown("---")

            st.write(f"### Step {step.get('step_number', index + 1)}: {step.get('step_name', 'Untitled Step')}")

            st.write(f"**Description:** {step.get('description', '')}")
            st.write(f"**Time Block:** {step.get('time_block', '')}")
            st.write(f"**Deadline:** {step.get('deadline', '')}")
            st.write(f"**Accountability Check:** {step.get('accountability_check', '')}")

            current_status = step.get("status", "Not Started")

            status = st.selectbox(
                "Status",
                options=["Not Started", "In Progress", "Completed", "Missed", "Skipped"],
                index=["Not Started", "In Progress", "Completed", "Missed", "Skipped"].index(current_status)
                if current_status in ["Not Started", "In Progress", "Completed", "Missed", "Skipped"]
                else 0,
                key=f"checkin_status_{index}"
            )

            notes = st.text_area(
                "Check-in note",
                value=step.get("notes", ""),
                placeholder="Example: I completed this today, I struggled with time, or I need to adjust this step.",
                key=f"checkin_notes_{index}"
            )

            step["status"] = status
            step["notes"] = notes

            updated_steps.append(step)

    st.session_state.plan["steps"] = updated_steps

    completed_count = sum(
        1 for step in updated_steps
        if step.get("status") == "Completed"
    )

    total_steps = len(updated_steps)

    st.markdown("---")
    st.subheader("Progress Summary")

    if total_steps > 0:
        progress = completed_count / total_steps
        st.progress(progress)
        st.write(f"{completed_count} out of {total_steps} steps completed.")
    else:
        st.write("No steps found.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save Check-In"):
            st.success("Check-in saved for this session.")

    with col2:
        if st.button("Back to Plan"):
            st.session_state.page = "goal_setup"
            st.rerun()

    st.download_button(
        label="Download Updated Plan",
        data=json.dumps(st.session_state.plan, indent=2),
        file_name="updated_accountability_plan.json",
        mime="application/json"
    )

    show_feedback_link("bottom")


# -----------------------------
# UI
# -----------------------------
init_session_state()

st.sidebar.title("Navigation")

selected_page = st.sidebar.radio(
    "Go to",
    ["Goal Setup", "Progress Tracker"],
    index=0 if st.session_state.page == "goal_setup" else 1
)

if selected_page == "Goal Setup":
    st.session_state.page = "goal_setup"
else:
    st.session_state.page = "daily_checkin"

if st.session_state.page == "daily_checkin":
    show_daily_checkin_page()
    st.stop()

st.set_page_config(
    page_title="Accountability Goal Tracker",
    page_icon="✅",
    layout="wide"
)

show_feedback_link("top")

st.title("✅ Accountability Goal Tracker")
st.write("Turn a vague goal into a SMART goal, break it into steps, and track your progress.")


# -----------------------------
# Step 1: User Goal
# -----------------------------
st.header("1. What goal are you trying to achieve?")

goal = st.text_area(
    "Enter your goal",
    value=st.session_state.goal,
    placeholder="Example: I want to lose weight, build my business, study better, or save money."
)

if st.button("Generate SMART Suggestions"):
    if not goal.strip():
        st.warning("Please enter a goal first.")
    else:
        st.session_state.goal = goal.strip()

        with st.spinner("Generating SMART suggestions..."):
            try:
                st.session_state.smart_suggestions = generate_smart_suggestions(st.session_state.goal)
                st.success("SMART suggestions generated.")
            except Exception as e:
                st.error(f"Could not generate SMART suggestions: {e}")


# -----------------------------
# Step 2: SMART Inputs
# -----------------------------
if st.session_state.smart_suggestions:
    st.header("2. Build Your SMART Goal")

def use_smart_suggestion(field_key):
    input_key = f"input_{field_key}"
    suggestion = st.session_state.smart_suggestions.get(field_key, "")

    st.session_state.smart_inputs[field_key] = suggestion
    st.session_state[input_key] = suggestion

for key, field in SMART_FIELDS.items():
    st.subheader(field["label"])

    st.caption(field["reason"])
    st.caption(field["principle"])

    suggestion = st.session_state.smart_suggestions.get(key, "")
    input_key = f"input_{key}"

    if input_key not in st.session_state:
        st.session_state[input_key] = st.session_state.smart_inputs.get(key, "")

    col1, col2 = st.columns([4, 1])

    with col1:
        st.text_area(
            f"Your answer for {field['label']}",
            placeholder=suggestion,
            key=input_key
        )

        st.session_state.smart_inputs[key] = st.session_state[input_key]

    with col2:
        st.write("")
        st.write("")
        st.button(
            "Use suggestion",
            key=f"use_{key}",
            on_click=use_smart_suggestion,
            args=(key,)
        )

    if suggestion:
        st.info(f"Suggested example: {suggestion}")

# -----------------------------
# Step 3: Generate Plan
# -----------------------------
if st.session_state.smart_suggestions:
    st.header("3. Generate Manageable Steps")

    if st.button("Generate Accountability Plan"):
        missing_fields = [
            SMART_FIELDS[key]["label"]
            for key, value in st.session_state.smart_inputs.items()
            if not value.strip()
        ]

        if missing_fields:
            st.warning(f"Please complete these SMART fields first: {', '.join(missing_fields)}")
        else:
            with st.spinner("Generating your step-by-step accountability plan..."):
                try:
                    st.session_state.plan = generate_plan(
                        st.session_state.goal,
                        st.session_state.smart_inputs
                    )
                    st.success("Plan generated.")
                except Exception as e:
                    st.error(f"Could not generate plan: {e}")


# -----------------------------
# Step 4: Display Plan
# -----------------------------
if st.session_state.plan:
    plan = st.session_state.plan

    st.header("4. Your Generated Plan")

    st.subheader("Goal Summary")
    st.write(plan.get("goal_summary", ""))

    st.subheader("Likely Obstacle")
    st.write(plan.get("likely_obstacle", ""))

    st.subheader("Today’s Next Action")
    st.success(plan.get("today_next_action", ""))

    st.subheader("Tracking Table")

    df = plan_to_dataframe(plan)

    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "status": st.column_config.SelectboxColumn(
                "status",
                options=["Not Started", "In Progress", "Completed", "Skipped"]
            )
        }
    )

    st.session_state.plan["steps"] = edited_df.to_dict(orient="records")

    st.download_button(
        label="Download Plan as JSON",
        data=json.dumps(st.session_state.plan, indent=2),
        file_name="accountability_plan.json",
        mime="application/json"
    )


# -----------------------------
# Step 5: Satisfaction Loop
# -----------------------------
if st.session_state.plan:
    st.header("5. Are you satisfied with this plan?")

    col_yes, col_no = st.columns(2)

    with col_yes:
         if st.button("Yes, I am satisfied"):
            st.session_state.satisfied = True
            st.success("Great! Your plan is ready. Go to the sidebar and click 'Progress Tracker' to start tracking your progress.")
            st.info("On the Progress Tracker page, you can update each step, add notes, and monitor your completion progress.")

    with col_no:
        if st.button("No, improve the plan"):
            st.session_state.satisfied = False

    if not st.session_state.satisfied:
        st.subheader("Tell me what needs to change")

        st.session_state.clarification_notes = st.text_area(
            "What is wrong with the plan? What obstacles, schedule issues, or missing details should I consider?",
            value=st.session_state.clarification_notes,
            placeholder="Example: The steps are too hard, I only have 30 minutes per day, I work evening shifts, or the deadline is too tight."
        )

        if st.button("Regenerate Improved Plan"):
            if not st.session_state.clarification_notes.strip():
                st.warning("Please explain what needs to change first.")
            else:
                with st.spinner("Regenerating the plan using your feedback and chat history..."):
                    try:
                        st.session_state.plan = generate_plan(
                            st.session_state.goal,
                            st.session_state.smart_inputs,
                            st.session_state.clarification_notes
                        )
                        st.success("Improved plan generated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not regenerate plan: {e}")


# -----------------------------
# Debug / Developer View
# -----------------------------

show_feedback_link("bottom")

with st.expander("Developer View: Stored Data Structure"):
    st.write("Goal:")
    st.json(st.session_state.goal)

    st.write("SMART Inputs:")
    st.json(st.session_state.smart_inputs)

    st.write("Plan:")
    st.json(st.session_state.plan)

    st.write("Chat History:")
    st.json(st.session_state.chat_history)