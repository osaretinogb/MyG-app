"""MyG Accountability App.

This Streamlit application helps a signed-in user turn a broad goal into a
SMART goal, generate an action plan with OpenAI, save the plan in Supabase,
and track progress over time.

The file is divided into clearly labelled sections so it is easier to learn
what each block does and where future changes should be made.
"""

# Python standard-library imports.
import base64
import hashlib
import json
import os
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List
from urllib.parse import unquote

# Third-party package imports.
import extra_streamlit_components as stx
import pandas as pd
import streamlit as st
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client


# Streamlit requires page configuration before the app creates visible UI.
st.set_page_config(
    page_title="Accountability Goal Tracker",
    page_icon="✅",
    layout="wide",
)



# -----------------------------
# Environment, secrets, and OpenAI setup
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

# Usage limits
MAX_PLAN_GENERATIONS_PER_DAY = 3
MAX_PLAN_REGENERATIONS_PER_DAY = 5
MAX_GOAL_CHARACTERS = 500
MAX_CLARIFICATION_CHARACTERS = 800

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
# Session State defaults
# -----------------------------
def init_session_state():
    """Create every Session State value used by the application.

    Streamlit reruns this file after most user interactions. These checks keep
    existing values instead of resetting them during an ordinary rerun.
    """
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

    # Default page used when the URL contains no valid `view` parameter.
    if "page" not in st.session_state:
        st.session_state.page = "my_plans"

    if "current_plan_id" not in st.session_state:
        st.session_state.current_plan_id = None

    if "plan_loaded_for_editing" not in st.session_state:
        st.session_state.plan_loaded_for_editing = False

    # Authentication objects exist only for the current Streamlit session.
    # A browser refresh rebuilds them from the encrypted authentication cookie.
    if "user" not in st.session_state:
        st.session_state.user = None

    if "session" not in st.session_state:
        st.session_state.session = None

    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "Login"






# -----------------------------
# Authentication and OpenAI helper functions
# -----------------------------

# Detect whether a Supabase exception represents an invalid login session.
def is_auth_session_error(error):
    """
    Returns True when an error appears to mean that the
    user's Supabase authentication session is no longer valid.
    """
    message = str(error).lower()

    auth_error_phrases = [
        "invalid refresh token",
        "refresh token already used",
        "refresh_token_already_used",
        "refresh_token_not_found",
        "jwt expired",
        "invalid jwt",
        "session expired",
        "not authenticated",
        "authentication required",
    ]

    return any(
        phrase in message
        for phrase in auth_error_phrases
    )


# Clear broken authentication state and safely return the user to login.
def return_user_to_login(message):
    """
    Clears the invalid local login session and redirects
    the user to the login page.
    """

    keys_to_clear = [
        "user",
        "session",
        "supabase_client",
        "current_plan_id",
        "plan",
        "user_plans",
        "page",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    # Set this after clearing the other values so it survives the rerun.
    st.session_state.auth_notice = message

    st.rerun()

#----------------------------------------------------------------------

# Authenticate the email/password form and store the resulting Supabase session.
def handle_login():
    email = st.session_state.get("login_email", "").strip().lower()
    password = st.session_state.get("login_password", "")

    if not email or not password:
        st.session_state.login_error = (
            "Please enter your email and password."
        )
        return

    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password,
        })

        if response.user and response.session:
            st.session_state.user = response.user
            st.session_state.session = response.session
            st.session_state.login_error = ""

            st.session_state.pop("logout_in_progress", None)

            save_auth_cookie(response.session)

            set_start_page_after_login()
        else:
            st.session_state.login_error = (
                "Login was unsuccessful. Check your email and password."
            )

    except Exception as error:
        st.session_state.login_error = f"Login failed: {error}"

#-----------------------------------------------------------------
# Render the login and account-creation forms.
def show_login_page():
    st.title("🔐 MyG Accountability App")
    st.caption("Public Beta")

    auth_notice = st.session_state.pop(
        "auth_notice",
        None
    )

    if auth_notice:
        st.warning(auth_notice)

    st.warning(
        "Public Beta: This app is still being improved. "
        "Please do not enter highly sensitive personal information. "
        "Use it for goal planning and accountability testing."
    )

    st.info(
        "Create an account or log in to save your goals, plans, and progress."
    )

    if supabase is None:
        st.error(
            "Supabase is not configured. "
            "Login cannot work until the Supabase keys are added."
        )
        st.stop()

    auth_mode = st.radio(
        "Choose an option",
        ["Login", "Sign Up"],
        horizontal=True,
        key="auth_mode_radio"
    )

    # -----------------------------
    # Login form
    # -----------------------------
    if auth_mode == "Login":
        if "login_error" not in st.session_state:
            st.session_state.login_error = ""

        with st.form(
            "login_form",
            clear_on_submit=False,
            enter_to_submit=True
        ):
            st.text_input(
                "Email",
                key="login_email",
                autocomplete="email"
            )

            st.text_input(
                "Password",
                type="password",
                key="login_password",
                autocomplete="current-password"
            )

            st.form_submit_button(
                "Login",
                use_container_width=True,
                type="primary",
                on_click=handle_login
            )

        if st.session_state.user is not None:
            st.rerun()

        if st.session_state.login_error:
            st.error(st.session_state.login_error)

    # -----------------------------
    # Sign-up form
    # -----------------------------
    else:
        with st.form(
            "signup_form",
            clear_on_submit=False,
            enter_to_submit=True
        ):
            email = st.text_input(
                "Email",
                key="signup_email",
                autocomplete="email"
            )

            password = st.text_input(
                "Password",
                type="password",
                key="signup_password",
                autocomplete="new-password"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                key="signup_confirm_password",
                autocomplete="new-password"
            )

            signup_submitted = st.form_submit_button(
                "Create Account",
                use_container_width=True,
                type="primary"
            )

        if signup_submitted:
            email = email.strip().lower()

            if not email or not password or not confirm_password:
                st.warning(
                    "Please enter your email, password, "
                    "and confirmation password."
                )

            elif password != confirm_password:
                st.warning("Passwords do not match.")

            elif len(password) < 6:
                st.warning(
                    "Password must contain at least 6 characters."
                )

            else:
                try:
                    response = supabase.auth.sign_up({
                        "email": email,
                        "password": password,
                    })

                    if response.user:
                        if response.session:
                            st.session_state.user = response.user
                            st.session_state.session = response.session

                            st.session_state.pop("logout_in_progress", None)

                            save_auth_cookie(response.session)
                            set_start_page_after_login()
                            st.rerun()

                        else:
                            st.success("Account created successfully.")
                            st.info(
                                "Check your email to confirm your account, "
                                "then return here to log in."
                            )

                    else:
                        st.error(
                            "Account creation did not complete. "
                            "Please try again."
                        )

                except Exception as error:
                    st.error(f"Sign up failed: {error}")
                    
#----------------------------------------------------

def logout_user():
    """
    Sign out from Supabase and prepare the browser cookie
    for deletion without accidentally restoring it again.
    """

    # Tell the next Streamlit rerun that this logout was intentional.
    st.session_state.logout_in_progress = True

    current_client = st.session_state.get("supabase_client")

    try:
        if current_client is not None:
            current_client.auth.sign_out()
    except Exception as error:
        # Even if remote sign-out fails, clear the local login.
        print(
            "LOGOUT WARNING:",
            type(error).__name__,
            str(error),
        )

    # Clear local authentication before the cookie component runs.
    keys_to_clear = [
        "user",
        "session",
        "supabase_client",
        "current_plan_id",
        "plan",
        "user_plans",
        "page",
        "auth_cookie_session_fingerprint",
        "pending_auth_cookie_session",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    # Delete the cookie during the next startup run.
    # This avoids a race with the browser cookie component.
    st.session_state.pending_auth_cookie_delete = True

    st.query_params.clear()

    st.session_state.auth_notice = (
        "You have been logged out."
    )

    st.rerun()
#-----------------------------------------------------

# Send the accumulated conversation to the OpenAI Responses API.
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

#------------------------------------------------------------

# Convert the model response into a Python dictionary.
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
# Ask the model for suggested values for each SMART field.
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


# Ask the model to turn the completed SMART goal into actionable steps.
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

#-----------------------------------------------

# Convert plan steps into a DataFrame for Streamlit's editable table.
def plan_to_dataframe(plan: Dict[str, Any]) -> pd.DataFrame:
    steps = plan.get("steps", [])
    return pd.DataFrame(steps)

#---------------------------------------------------------

# Display the Google Form feedback link at the top or bottom of a page.
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

#---------------------------------------------------------
    
# Render the Progress Tracker and allow users to update step progress.
def show_daily_checkin_page():

    show_feedback_link("top")
    
    st.title("📅 Daily Check-In Page")
    st.info("You are now on the Daily Check-In page. Update your progress below.")

    if not st.session_state.plan:
        st.warning("No plan found yet. Please create a goal plan first.")

        if st.button("Go back to goal setup"):
            set_current_page("goal_setup")
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
            if st.session_state.current_plan_id:
                update_step_progress_in_supabase(
                    st.session_state.current_plan_id,
                    st.session_state.plan["steps"]
                )
                st.success("Check-in saved permanently.")
            else:
                st.warning("This plan has not been saved to Supabase yet. Your check-in is saved only for this session.")

    with col2:
        if st.button("Back to Plan"):
            set_current_page("goal_setup")
            st.rerun()

    st.download_button(
        label="Download Updated Plan",
        data=json.dumps(st.session_state.plan, indent=2),
        file_name="updated_accountability_plan.json",
        mime="application/json"
    )

    show_feedback_link("bottom")

#-----------------------------------------------------------

# Read a secret from Streamlit Cloud first, then fall back to the local .env file.
def get_secret_value(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

#-----------------------------------------------------------

supabase_url = get_secret_value("SUPABASE_URL")
supabase_key = get_secret_value("SUPABASE_ANON_KEY")

if not supabase_url or not supabase_key:
    st.error(
        "Supabase is not configured. "
        "Add SUPABASE_URL and SUPABASE_ANON_KEY."
    )
    st.stop()
#----------------------------------------------------------

# Create one Supabase client per Streamlit session and reuse it on reruns.
def get_supabase_client():
    """
    Creates one Supabase client for the current
    Streamlit user session and reuses it on reruns.
    """

    if "supabase_client" not in st.session_state:
        st.session_state.supabase_client = create_client(
            supabase_url,
            supabase_key
        )

    return st.session_state.supabase_client

#----------------------------------------------------------

# Insert a plan and all of its steps into Supabase.
def save_plan_to_supabase(plan, user_id):
    try:
        if supabase is None:
            st.error("Supabase is not configured, so the plan could not be saved.")
            return None

        plan_payload = {
            "user_id": user_id,
            "goal_summary": plan.get("goal_summary", ""),
            "smart_goal": plan.get("smart_goal", {}),
            "overall_deadline": plan.get("overall_deadline", ""),
            "likely_obstacle": plan.get("likely_obstacle", ""),
            "today_next_action": plan.get("today_next_action", "")
        }

        plan_response = supabase.table("plans").insert(plan_payload).execute()

        if not plan_response.data:
            st.error("Plan could not be saved.")
            return None

        plan_id = plan_response.data[0]["id"]

        step_rows = []

        for step in plan.get("steps", []):
            step_rows.append({
                "plan_id": plan_id,
                "step_number": step.get("step_number"),
                "step_name": step.get("step_name", ""),
                "description": step.get("description", ""),
                "time_block": step.get("time_block", ""),
                "deadline": step.get("deadline", ""),
                "accountability_check": step.get("accountability_check", ""),
                "status": step.get("status", "Not Started"),
                "notes": step.get("notes", "")
            })

        if step_rows:
            supabase.table("plan_steps").insert(step_rows).execute()

        return plan_id

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. Please log in again. "
                "Any changes that were successfully autosaved remain available."
            )

        st.error(f"Could not save your plan: {error}")
        return None

#-------------------------------------------------------

# Retrieve one saved plan and rebuild the app's plan dictionary.
def load_plan_from_supabase(plan_id):
    if supabase is None:
        st.error("Supabase is not configured.")
        return None
    
    try:

        plan_response = (
            supabase.table("plans")
            .select("*")
            .eq("id", plan_id)
            .single()
            .execute()
        )

        if not plan_response.data:
            st.error("No plan found with that Plan ID.")
            return None

        steps_response = (
            supabase.table("plan_steps")
            .select("*")
            .eq("plan_id", plan_id)
            .order("step_number")
            .execute()
        )

        plan_data = plan_response.data
        steps_data = steps_response.data or []

        return {
            "goal_summary": plan_data.get("goal_summary", ""),
            "smart_goal": plan_data.get("smart_goal", {}),
            "overall_deadline": plan_data.get("overall_deadline", ""),
            "likely_obstacle": plan_data.get("likely_obstacle", ""),
            "today_next_action": plan_data.get("today_next_action", ""),
            "steps": [
                {
                    "step_number": step.get("step_number"),
                    "step_name": step.get("step_name", ""),
                    "description": step.get("description", ""),
                    "time_block": step.get("time_block", ""),
                    "deadline": step.get("deadline", ""),
                    "accountability_check": step.get("accountability_check", ""),
                    "status": step.get("status", "Not Started"),
                    "notes": step.get("notes", "")
                }
                for step in steps_data
            ]
        }

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. Please log in again. "
                "Any changes that were successfully autosaved remain available."
            )

        st.session_state.autosave_error = str(error)
        return False
#-------------------------------------------------------

# Save status and notes for every step in the Progress Tracker.
def update_step_progress_in_supabase(plan_id, steps):
    if supabase is None:
        st.warning("Supabase is not configured. Progress was saved only for this session.")
        return

    for step in steps:
        step_number = step.get("step_number")

        supabase.table("plan_steps").update({
            "status": step.get("status", "Not Started"),
            "notes": step.get("notes", "")
        }).eq("plan_id", plan_id).eq("step_number", step_number).execute()


#---------------------------------------------------

# Delete a plan; the database cascade should remove its steps.
def delete_plan_from_supabase(plan_id):
    if supabase is None:
        st.error("Supabase is not configured.")
        return False

    try:
        supabase.table("plans").delete().eq("id", plan_id).execute()
        return True
    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. Please log in again. "
                "Please Log in again."
            )

        st.error(f"Could not delete your plans: {error}")
        return False

#--------------------------------------------------

# Update a plan and replace its saved step rows.
def update_existing_plan_in_supabase(plan_id, plan):
    if supabase is None:
        st.error("Supabase is not configured.")
        return False

    try:
        plan_payload = {
            "goal_summary": plan.get("goal_summary", ""),
            "smart_goal": plan.get("smart_goal", {}),
            "overall_deadline": plan.get("overall_deadline", ""),
            "likely_obstacle": plan.get("likely_obstacle", ""),
            "today_next_action": plan.get("today_next_action", "")
        }

        supabase.table("plans").update(plan_payload).eq("id", plan_id).execute()

        # Simple MVP approach:
        # Delete old steps, then reinsert current edited steps.
        supabase.table("plan_steps").delete().eq("plan_id", plan_id).execute()

        step_rows = []

        for step in plan.get("steps", []):
            step_rows.append({
                "plan_id": plan_id,
                "step_number": step.get("step_number"),
                "step_name": step.get("step_name", ""),
                "description": step.get("description", ""),
                "time_block": step.get("time_block", ""),
                "deadline": step.get("deadline", ""),
                "accountability_check": step.get("accountability_check", ""),
                "status": step.get("status", "Not Started"),
                "notes": step.get("notes", "")
            })

        if step_rows:
            supabase.table("plan_steps").insert(step_rows).execute()

        return True

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. Please log in again. "
                "Any changes that were successfully autosaved remain available."
            )

        st.error(f"Could not load your plans: {error}")
        return False

#---------------------------------------------------
# Persistent authentication: encrypted browser-cookie management
#------------------------------------------------
AUTH_COOKIE_NAME = "myg_auth_session"

cookie_secret = get_secret_value("COOKIE_SECRET")

if not cookie_secret:
    st.error(
        "COOKIE_SECRET is missing. Add it to your local .env file "
        "and Streamlit Cloud secrets."
    )
    st.stop()


# Derive a Fernet encryption key from COOKIE_SECRET.
def create_cookie_cipher():
    """
    Converts COOKIE_SECRET into a valid Fernet encryption key.
    """
    secret_hash = hashlib.sha256(
        cookie_secret.encode("utf-8")
    ).digest()

    fernet_key = base64.urlsafe_b64encode(secret_hash)

    return Fernet(fernet_key)


cookie_cipher = create_cookie_cipher()


# The cookie manager is initialized later, after login restoration.
# Creating this browser component before restoration can trigger an extra
# Streamlit rerun and interrupt Supabase before the user is restored.
cookie_manager = None

#------------------------------------------------
# Add cookie helper functions to manage the authentication session data.
#------------------------------------------------

# Use HTTPS-only cookies in production but allow HTTP on localhost.
def should_use_secure_cookie():
    """
    Returns False for local Mac development and True
    for the deployed HTTPS application.
    """
    try:
        host = (
            st.context.headers
            .get("host", "")
            .split(":")[0]
            .lower()
        )

        if not host:
            return False

        local_hosts = {
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "::1",
        }

        return host not in local_hosts

    except Exception:
        return False

def save_auth_cookie(session):
    """Encrypt and save the newest Supabase token pair in a browser cookie.

    Supabase can rotate refresh tokens. The fingerprint prevents unnecessary
    cookie writes while still allowing a newly rotated token pair to replace
    the old cookie.
    """
    if not session:
        return

    access_token = getattr(session, "access_token", "")
    refresh_token = getattr(session, "refresh_token", "")

    if not access_token or not refresh_token:
        return

    session_fingerprint = hashlib.sha256(
        f"{access_token}:{refresh_token}".encode("utf-8")
    ).hexdigest()

    if (
        st.session_state.get("auth_cookie_session_fingerprint")
        == session_fingerprint
    ):
        return

    session_payload = {
        "access_token": access_token,
        "refresh_token": refresh_token,
    }

    encrypted_value = cookie_cipher.encrypt(
        json.dumps(session_payload).encode("utf-8")
    ).decode("utf-8")

    # A changing component key lets Streamlit execute another cookie update
    # when Supabase supplies a different token pair.
    component_key = session_fingerprint[:12]

    if cookie_manager is None:
        raise RuntimeError("Cookie manager has not been initialized.")

    # Record the fingerprint before calling the browser component. The
    # component can request a Streamlit rerun, so this prevents a write loop.
    st.session_state.auth_cookie_session_fingerprint = session_fingerprint

    cookie_manager.set(
        AUTH_COOKIE_NAME,
        encrypted_value,
        key=f"set_auth_cookie_{component_key}",
        path="/",
        expires_at=datetime.now() + timedelta(days=30),
        secure=should_use_secure_cookie(),
        same_site="strict",
    )


def delete_auth_cookie():
    """Expire the persistent authentication cookie in the browser."""
    if cookie_manager is None:
        # Restoration runs before the browser cookie component is created.
        # Defer deletion until startup initializes the component.
        st.session_state.pending_auth_cookie_delete = True
        return

    st.session_state.pop("auth_cookie_session_fingerprint", None)

    cookie_manager.set(
        AUTH_COOKIE_NAME,
        "",
        key="clear_myg_auth_cookie",
        path="/",
        expires_at=datetime.now() - timedelta(days=1),
        secure=should_use_secure_cookie(),
        same_site="strict",
    )

#------------------------------------------------
# Login restoration function
#------------------------------------------------

def restore_login_from_cookie():
    """Rebuild the Supabase login after a full browser refresh.

    The function reads the encrypted token pair from the request cookie,
    restores it into the new Supabase client, verifies the user, and saves any
    rotated tokens back to the cookie.
    """
    if st.session_state.get("user") is not None:
        return True

    raw_cookie = st.context.cookies.get(AUTH_COOKIE_NAME)

    if not raw_cookie:
        return False

    # Some browsers/components may quote or URL-encode a cookie value.
    encrypted_cookie = unquote(str(raw_cookie)).strip()
    if (
        len(encrypted_cookie) >= 2
        and encrypted_cookie[0] == encrypted_cookie[-1]
        and encrypted_cookie[0] in {'"', "'"}
    ):
        encrypted_cookie = encrypted_cookie[1:-1]

    try:
        decrypted_value = cookie_cipher.decrypt(
            encrypted_cookie.encode("utf-8")
        ).decode("utf-8")

        saved_tokens = json.loads(decrypted_value)
        access_token = saved_tokens.get("access_token")
        refresh_token = saved_tokens.get("refresh_token")

        if not access_token or not refresh_token:
            raise ValueError("Saved login tokens are incomplete.")

        # Mark the token pair already stored in the browser. This prevents the
        # later synchronization step from rewriting an unchanged cookie.
        saved_session_fingerprint = hashlib.sha256(
            f"{access_token}:{refresh_token}".encode("utf-8")
        ).hexdigest()
        st.session_state.auth_cookie_session_fingerprint = (
            saved_session_fingerprint
        )

        # set_session restores a valid session and refreshes it when needed.
        response = supabase.auth.set_session(access_token, refresh_token)
        restored_session = getattr(response, "session", None)
        restored_user = getattr(response, "user", None)

        if restored_session is None:
            raise ValueError("Supabase returned no restored session.")

        # Some client versions may not populate response.user consistently.
        # Verify the access token directly before trusting the restored login.
        if restored_user is None:
            user_response = supabase.auth.get_user(
                restored_session.access_token
            )
            restored_user = getattr(user_response, "user", None)

        if restored_user is None:
            raise ValueError("Supabase returned no restored user.")

        st.session_state.user = restored_user
        st.session_state.session = restored_session
        st.session_state.login_error = ""

        # If Supabase rotated either token, postpone the cookie write until
        # after the browser cookie component is initialized. Authentication is
        # already restored, so a component-triggered rerun cannot log us out.
        tokens_changed = (
            restored_session.access_token != access_token
            or restored_session.refresh_token != refresh_token
        )

        if tokens_changed:
            st.session_state.pending_auth_cookie_session = restored_session

        return True

    except (InvalidToken, json.JSONDecodeError, ValueError) as error:
        # These errors indicate a malformed, stale, or unreadable local cookie.
        print(
            "AUTH RESTORE ERROR:",
            type(error).__name__,
            str(error),
        )
        delete_auth_cookie()
        st.session_state.auth_notice = (
            "Your saved login was no longer valid. Please log in again."
        )
        return False

    except Exception as error:
        # Do not print the cookie or either token. Only print the error type and
        # message so authentication can be diagnosed without exposing secrets.
        print(
            "AUTH RESTORE ERROR:",
            type(error).__name__,
            str(error),
        )

        if is_auth_session_error(error):
            delete_auth_cookie()
            st.session_state.auth_notice = (
                "Your login session expired. Please log in again."
            )
        else:
            # Keep the cookie for an unexpected temporary/network error.
            st.session_state.auth_notice = (
                "Your login could not be restored. Please try again."
            )

        return False


def sync_auth_cookie_from_supabase():
    """Keep the browser cookie synchronized with Supabase's current session.

    Supabase refresh tokens are rotated. Calling get_session during an ordinary
    app rerun lets Supabase refresh an expired access token, and save_auth_cookie
    stores the newest token pair before a later browser refresh occurs.
    """
    if st.session_state.get("user") is None:
        return False

    try:
        current_session = supabase.auth.get_session()

        if current_session is None:
            return_user_to_login(
                "Your login session expired. Please log in again."
            )

        st.session_state.session = current_session
        save_auth_cookie(current_session)
        return True

    except Exception as error:
        if is_auth_session_error(error):
            delete_auth_cookie()
            return_user_to_login(
                "Your login session expired. Please log in again."
            )

        # A temporary network problem should not automatically sign the user
        # out. Database operations can show their own error when attempted.
        print(
            "AUTH COOKIE SYNC ERROR:",
            type(error).__name__,
            str(error),
        )
        return False
    
#-------------------------------------------------
# Retrieve the signed-in user's saved plans for the My Plans page.
def get_user_plans(user_id):
    if supabase is None:
        return []
    try:
        response = (
            supabase.table("plans")
            .select("id, created_at, goal_summary, overall_deadline, likely_obstacle, today_next_action")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        return response.data or []
    
    
    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. "
                "Please log in again to access your plans."
            )

        st.error(f"Could not load your plans: {error}")
        return []

#------------------------------------------

# Render saved plans and actions to open, modify, or delete them.
def show_my_plans_page():
    show_feedback_link("top")

    st.title("📋 My Plans")
    st.write("Manage your saved accountability plans or create a new one.")

    user_id = st.session_state.user.id
    user_plans = get_user_plans(user_id)

    if not user_plans:
        st.info("You do not have any saved plans yet. Create your first plan to get started.")

        if st.button("Create My First Plan"):
            set_current_page("goal_setup")
            st.session_state.plan = None
            set_current_plan_id(None)
            st.session_state.smart_suggestions = {}
            st.session_state.smart_inputs = {
                "specific": "",
                "measurable": "",
                "achievable": "",
                "relevant": "",
                "time_bound": "",
            }
            st.rerun()

        show_feedback_link("bottom")
        return

    col_new, col_refresh = st.columns(2)

    with col_new:
        if st.button("➕ Create New Plan"):
            set_current_page("goal_setup")
            st.session_state.plan = None
            set_current_plan_id(None)
            st.session_state.goal = ""
            st.session_state.smart_suggestions = {}
            st.session_state.smart_inputs = {
                "specific": "",
                "measurable": "",
                "achievable": "",
                "relevant": "",
                "time_bound": "",
            }
            st.rerun()

    with col_refresh:
        if st.button("🔄 Refresh Plans"):
            st.rerun()

    st.markdown("---")

    for plan in user_plans:
        with st.container():
            st.subheader(plan.get("goal_summary", "Untitled Plan"))

            st.write(f"**Deadline:** {plan.get('overall_deadline', 'No deadline')}")
            st.write(f"**Next Action:** {plan.get('today_next_action', 'No next action')}")
            st.caption(f"Created: {plan.get('created_at', '')}")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Open Tracker", key=f"open_{plan['id']}"):
                    loaded_plan = load_plan_from_supabase(plan["id"])

                    if loaded_plan:
                        st.session_state.plan = loaded_plan
                        set_current_plan_id(plan["id"])
                        set_current_page("daily_checkin")
                        st.rerun()

            with col2:
                if st.button("Modify Plan", key=f"modify_{plan['id']}"):
                    loaded_plan = load_plan_from_supabase(plan["id"])

                    if loaded_plan:
                        st.session_state.plan = loaded_plan
                        set_current_plan_id(plan["id"])
                        st.session_state.plan_loaded_for_editing = True
                        set_current_page("goal_setup")
                        st.rerun()

            with col3:
                confirm_delete = st.checkbox(
                    "Confirm delete",
                    key=f"confirm_delete_{plan['id']}"
                )

                if st.button("Delete", key=f"delete_{plan['id']}"):
                    if not confirm_delete:
                        st.warning("Please check 'Confirm delete' first.")
                    else:
                        deleted = delete_plan_from_supabase(plan["id"])

                        if deleted:
                            if st.session_state.current_plan_id == plan["id"]:
                                set_current_plan_id(None)
                                st.session_state.plan = None

                            st.success("Plan deleted.")
                            st.rerun()

            st.markdown("---")

    show_feedback_link("bottom")

#-------------------------------------------

# Send new users to Goal Setup and returning users to My Plans.
def set_start_page_after_login():
    if st.session_state.user is None:
        return

    user_plans = get_user_plans(st.session_state.user.id)

    if user_plans:
        set_current_page("my_plans")
    else:
        set_current_page("goal_setup")

#-------------------------------------------------

# Read or create today's API-usage counter row.
def get_today_usage(user_id):
    """
    Return today's usage record for the logged-in user.

    If no record exists, create it safely. The upsert uses the
    unique combination of user_id and usage_date, preventing
    duplicate-row errors when Streamlit reruns simultaneously.
    """

    default_usage = {
        "plan_generations": 0,
        "plan_regenerations": 0,
    }

    if supabase is None:
        return default_usage

    today = date.today().isoformat()

    try:
        # First, check whether today's row already exists.
        existing_response = (
            supabase.table("usage_limits")
            .select("*")
            .eq("user_id", user_id)
            .eq("usage_date", today)
            .execute()
        )

        if existing_response.data:
            return existing_response.data[0]

        # Safely attempt to create today's row.
        # If another rerun creates it first, ignore the duplicate.
        (
            supabase.table("usage_limits")
            .upsert(
                {
                    "user_id": user_id,
                    "usage_date": today,
                    "plan_generations": 0,
                    "plan_regenerations": 0,
                },
                on_conflict="user_id,usage_date",
                ignore_duplicates=True,
            )
            .execute()
        )

        # Read the row again, regardless of which rerun created it.
        final_response = (
            supabase.table("usage_limits")
            .select("*")
            .eq("user_id", user_id)
            .eq("usage_date", today)
            .single()
            .execute()
        )

        if final_response.data:
            return final_response.data

        return default_usage

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. "
                "Please log in again."
            )

        st.error(
            f"Could not load today's usage: {error}"
        )
        return default_usage

#-------------------------------------------------

# Check whether the user has remaining initial plan generations today.
def can_generate_plan(user_id):
    usage = get_today_usage(user_id)
    return usage.get("plan_generations", 0) < MAX_PLAN_GENERATIONS_PER_DAY

#-------------------------------------------------

# Check whether the user has remaining plan regenerations today.
def can_regenerate_plan(user_id):
    usage = get_today_usage(user_id)
    return usage.get("plan_regenerations", 0) < MAX_PLAN_REGENERATIONS_PER_DAY

#-------------------------------------------------

# Increase the appropriate daily usage counter after a successful API call.
def increment_usage(user_id, usage_type):
    if supabase is None:
        return

    usage = get_today_usage(user_id)
    usage_id = usage["id"]

    if usage_type == "plan_generation":
        new_count = usage.get("plan_generations", 0) + 1

        supabase.table("usage_limits").update({
            "plan_generations": new_count,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", usage_id).execute()

    elif usage_type == "plan_regeneration":
        new_count = usage.get("plan_regenerations", 0) + 1

        supabase.table("usage_limits").update({
            "plan_regenerations": new_count,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", usage_id).execute()

VALID_PAGES = {
    "my_plans",
    "goal_setup",
    "daily_checkin",
}


def set_current_page(page_name):
    """Update the app's page in Session State and in the browser URL."""
    if page_name not in VALID_PAGES:
        page_name = "my_plans"

    st.session_state.page = page_name
    st.query_params["view"] = page_name


def set_current_plan_id(plan_id):
    """Remember which plan is open so it can be reloaded after a refresh."""
    st.session_state.current_plan_id = plan_id

    if plan_id:
        st.query_params["plan_id"] = str(plan_id)
    elif "plan_id" in st.query_params:
        del st.query_params["plan_id"]

# -----------------------------
# Main application startup and page routing
# -----------------------------

# -----------------------------
# Initialize app state and services
# -----------------------------
# Session State handles ordinary Streamlit reruns. The encrypted cookie below
# rebuilds authentication after a complete browser refresh.
init_session_state()
supabase = get_supabase_client()

st.caption(
    f"Auth cookie detected: "
    f"{AUTH_COOKIE_NAME in st.context.cookies}"
)

# Restore Supabase before creating the third-party cookie component. The
# component can trigger an extra Streamlit rerun when the page first loads.
if (
    st.session_state.user is None
    and not st.session_state.get("logout_in_progress", False)
):
    restore_login_from_cookie()

# The cookie manager is needed for login, logout, and token updates. At this
# point any rerun it triggers is safe because restoration has already finished.
cookie_manager = stx.CookieManager(
    key="myg_cookie_manager"
)

# Complete any cookie operation that restoration had to postpone.
if st.session_state.pop("pending_auth_cookie_delete", False):
    delete_auth_cookie()

pending_cookie_session = st.session_state.pop(
    "pending_auth_cookie_session",
    None,
)
if pending_cookie_session is not None:
    save_auth_cookie(pending_cookie_session)

if st.session_state.user is None:
    show_login_page()
    st.stop()

# Keep the browser cookie updated if Supabase rotates tokens later.
sync_auth_cookie_from_supabase()


# -----------------------------
# Restore navigation from URL query parameters
# -----------------------------
requested_page = st.query_params.get("view")
requested_plan_id = st.query_params.get("plan_id")

if requested_page in VALID_PAGES:
    st.session_state.page = requested_page

# A full refresh clears st.session_state.plan. When the URL identifies a saved
# plan, load it again so Progress Tracker behaves like a normal page refresh.
if requested_plan_id:
    st.session_state.current_plan_id = requested_plan_id

    if (
        requested_page == "daily_checkin"
        and st.session_state.plan is None
    ):
        refreshed_plan = load_plan_from_supabase(requested_plan_id)

        if refreshed_plan:
            st.session_state.plan = refreshed_plan
        else:
            set_current_plan_id(None)
            set_current_page("my_plans")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

st.sidebar.success(
    f"Logged in as: {st.session_state.user.email}"
)

if st.sidebar.button("Logout"):
    logout_user()


selected_page = st.sidebar.radio(
    "Go to",
    ["My Plans", "Create New Plan", "Progress Tracker"],
    index=(
        0 if st.session_state.page == "my_plans"
        else 1 if st.session_state.page == "goal_setup"
        else 2
    )
)


# Update both session state and the URL.
if selected_page == "My Plans":
    set_current_page("my_plans")

elif selected_page == "Create New Plan":
    set_current_page("goal_setup")

else:
    set_current_page("daily_checkin")


# -----------------------------
# Daily usage
# -----------------------------
usage = get_today_usage(st.session_state.user.id)

remaining_generations = (
    MAX_PLAN_GENERATIONS_PER_DAY
    - usage.get("plan_generations", 0)
)

remaining_regenerations = (
    MAX_PLAN_REGENERATIONS_PER_DAY
    - usage.get("plan_regenerations", 0)
)

st.sidebar.markdown("---")
st.sidebar.subheader("Today's Usage")

st.sidebar.write(
    f"Plan generations left: "
    f"{max(0, remaining_generations)}"
)

st.sidebar.write(
    f"Regenerations left: "
    f"{max(0, remaining_regenerations)}"
)


# -----------------------------
# Page routing
# -----------------------------
if st.session_state.page == "my_plans":
    show_my_plans_page()
    st.stop()

if st.session_state.page == "daily_checkin":
    show_daily_checkin_page()
    st.stop()


# -----------------------------
# Goal Setup page
# -----------------------------
show_feedback_link("top")

st.title("✅ Accountability Goal Tracker")

st.write(
    "Turn a vague goal into a SMART goal, "
    "break it into steps, and track your progress."
)

# -----------------------------
# Step 1: User Goal
# -----------------------------
st.header("1. What goal are you trying to achieve?")

goal = st.text_area(
    "Enter your goal",
    value=st.session_state.goal,
    placeholder="Example: I want to lose weight, build my business, study better, or save money.",
    max_chars= MAX_GOAL_CHARACTERS
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

# Copy one AI suggestion into its editable SMART input widget.
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
        user_id = st.session_state.user.id

        if not can_generate_plan(user_id):
            st.error(
                f"You have reached today's limit of {MAX_PLAN_GENERATIONS_PER_DAY} plan generations. Please try again tomorrow."
            )
        else:
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
                        increment_usage(user_id, "plan_generation")
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

    if st.session_state.current_plan_id:
        if st.button("Save Changes to This Plan"):
            updated = update_existing_plan_in_supabase(
                st.session_state.current_plan_id,
                st.session_state.plan
            )

            if updated:
                st.success("Changes saved successfully.")
    else:
        st.info("This is a new plan. Click Save Plan to store it permanently.") 

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
            placeholder="Example: The steps are too hard, I only have 30 minutes per day, I work evening shifts, or the deadline is too tight.",
            max_chars=MAX_CLARIFICATION_CHARACTERS
        )

        if st.button("Regenerate Improved Plan"):
            user_id = st.session_state.user.id

            if not can_regenerate_plan(user_id):
                st.error(
                    f"You have reached today's limit of {MAX_PLAN_REGENERATIONS_PER_DAY} plan regenerations. Please try again tomorrow."
                )
            elif not st.session_state.clarification_notes.strip():
                st.warning("Please explain what needs to change first.")
            else:
                with st.spinner("Regenerating the plan using your feedback and chat history..."):
                    try:
                        st.session_state.plan = generate_plan(
                            st.session_state.goal,
                            st.session_state.smart_inputs,
                            st.session_state.clarification_notes
                        )
                        increment_usage(user_id, "plan_regeneration")
                        st.success("Improved plan generated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not regenerate plan: {e}")

st.subheader("Save Your Plan")

if st.session_state.current_plan_id:
    st.info("This plan is already saved. Use 'Save Changes to This Plan' after editing.")
else:
    if st.button("Save Plan"):
        user_id = st.session_state.user.id

        plan_id = save_plan_to_supabase(
            st.session_state.plan,
            user_id
        )

        if plan_id:
            set_current_plan_id(plan_id)
            st.success("Plan saved successfully.")
            st.code(plan_id)


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