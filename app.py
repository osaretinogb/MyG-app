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

    if "auth_check_complete" not in st.session_state:
        st.session_state.auth_check_complete = False

    if "auth_restore_attempts" not in st.session_state:
        st.session_state.auth_restore_attempts = 0

    if "autosave_message" not in st.session_state:
        st.session_state.autosave_message = ""

    if "autosave_error" not in st.session_state:
        st.session_state.autosave_error = ""






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

            st.session_state.auth_restore_attempts = 0
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

                            st.session_state.auth_restore_attempts = 0
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

    if st.session_state.get("autosave_message"):
        st.success(st.session_state.autosave_message)

    if st.session_state.get("autosave_error"):
        st.warning(
            f"Autosave failed: {st.session_state.autosave_error}"
        )

    

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

            status_options = [
                "Not Started",
                "In Progress",
                "Completed",
                "Missed",
                "Skipped"
            ]

            if current_status not in status_options:
                current_status = "Not Started"

            step_number = step.get("step_number", index + 1)

            # Include the plan ID so widget keys stay unique across different plans.
            plan_key = st.session_state.current_plan_id or "unsaved"

            status_key = f"checkin_status_{plan_key}_{step_number}"
            notes_key = f"checkin_notes_{plan_key}_{step_number}"

            # Initialize the widgets from the currently loaded plan data.
            if status_key not in st.session_state:
                st.session_state[status_key] = current_status

            if notes_key not in st.session_state:
                st.session_state[notes_key] = step.get("notes", "")

            status = st.selectbox(
                "Status",
                options=status_options,
                key=status_key,
                on_change=autosave_checkin,
                args=(index, status_key, notes_key)
            )

            notes = st.text_area(
                "Check-in note",
                placeholder=(
                    "Example: I completed this today, "
                    "I struggled with time, or I need to adjust this step."
                ),
                key=notes_key,
                on_change=autosave_checkin,
                args=(index, status_key, notes_key)
            )

            # Keep the current in-memory plan synchronized with the widgets.
            step["status"] = status
            step["notes"] = notes

            updated_steps.append(step)
            # ---------------------------------
            # Accountability partner comments
            # ---------------------------------
            step_id = step.get("id")

            st.write("DEBUG STEP ID:", step_id)

            if step_id:
                comments = get_step_comments(step_id)

                if comments:
                    st.markdown("#### 💬 Accountability")

                    for comment in comments:
                        author = comment.get("author") or {}

                        author_name = (
                            author.get("display_name")
                            or author.get("username")
                            or "MyG User"
                        )

                        st.write(f"**{author_name}**")
                        st.write(comment.get("comment_text", ""))

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
                    "id": step.get("id"),
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

#---------------------------------------------------

# Find another MyG user by exact username.

def search_profile_by_username(username):

    try:
        cleaned_username = username.strip().lower()

        response = (
            supabase.table("profiles")
            .select(
                "id, username, display_name, bio, avatar_url"
            )
            .eq("username", cleaned_username)
            .execute()
        )

        if response.data:
            return response.data[0]

        return None

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. "
                "Please log in again."
            )

        st.error(
            f"Could not search for user: {error}"
        )
        return None

#---------------------------------------------------

# Check both directions so users cannot create
# duplicate/reversed accountability relationships.

def get_existing_partnership(user_a_id, user_b_id):
    """
    Check both directions so users cannot create
    duplicate/reversed accountability relationships.
    """

    try:
        response = (
            supabase.table(
                "accountability_partnerships"
            )
            .select("*")
            .or_(
                f"and(requester_id.eq.{user_a_id},"
                f"recipient_id.eq.{user_b_id}),"
                f"and(requester_id.eq.{user_b_id},"
                f"recipient_id.eq.{user_a_id})"
            )
            .execute()
        )

        if response.data:
            return response.data[0]

        return None

    except Exception as error:
        st.error(
            f"Could not check partnership: {error}"
        )
        return None
    
#---------------------------------------------------

# Send an accountability partner invitation.

def send_partnership_request(recipient_id):
    """
    Send an accountability partner invitation.
    """

    requester_id = st.session_state.user.id

    if requester_id == recipient_id:
        st.warning(
            "You cannot invite yourself as an "
            "accountability partner."
        )
        return False

    existing = get_existing_partnership(
        requester_id,
        recipient_id
    )

    if existing:
        status = existing.get("status")

        if status == "accepted":
            st.info(
                "You are already accountability partners."
            )

        elif status == "pending":
            st.info(
                "There is already a pending request "
                "between you and this user."
            )

        elif status == "declined":
            st.info(
                "A previous accountability request "
                "between you and this user was declined."
            )

        return False

    try:
        response = (
            supabase.table(
                "accountability_partnerships"
            )
            .insert({
                "requester_id": requester_id,
                "recipient_id": recipient_id,
                "status": "pending",
            })
            .execute()
        )

        return bool(response.data)

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. "
                "Please log in again."
            )

        st.error(
            f"Could not send request: {error}"
        )
        return False

#---------------------------------------------------

# Get pending invitations sent to the logged-in user.

def get_incoming_partner_requests():
    """
    Get pending invitations sent to the logged-in user.
    """

    user_id = st.session_state.user.id

    try:
        response = (
            supabase.table(
                "accountability_partnerships"
            )
            .select(
                """
                id,
                requester_id,
                recipient_id,
                status,
                created_at,
                requester:profiles!accountability_partnerships_requester_id_fkey(
                    username,
                    display_name,
                    bio
                )
                """
            )
            .eq("recipient_id", user_id)
            .eq("status", "pending")
            .order("created_at", desc=True)
            .execute()
        )

        return response.data or []

    except Exception as error:
        st.error(
            f"Could not load requests: {error}"
        )
        return []

#---------------------------------------------------

# Accept or decline an accountability invitation.

def respond_to_partner_request(
    partnership_id,
    new_status
):

    if new_status not in [
        "accepted",
        "declined"
    ]:
        return False

    try:
        response = (
            supabase.table(
                "accountability_partnerships"
            )
            .update({
                "status": new_status,
                "updated_at": datetime.utcnow().isoformat(),
            })
            .eq("id", partnership_id)
            .eq(
                "recipient_id",
                st.session_state.user.id
            )
            .execute()
        )

        return bool(response.data)

    except Exception as error:
        st.error(
            f"Could not update request: {error}"
        )
        return False

#---------------------------------------------------

# Return all accepted partnerships for the current user.

def get_accountability_partners():

    user_id = st.session_state.user.id

    try:
        response = (
            supabase.table(
                "accountability_partnerships"
            )
            .select("*")
            .eq("status", "accepted")
            .or_(
                f"requester_id.eq.{user_id},"
                f"recipient_id.eq.{user_id}"
            )
            .execute()
        )

        partnerships = response.data or []
        partners = []

        for relationship in partnerships:
            if relationship["requester_id"] == user_id:
                partner_id = relationship["recipient_id"]
            else:
                partner_id = relationship["requester_id"]

            profile_response = (
                supabase.table("profiles")
                .select(
                    "id, username, display_name, bio, avatar_url"
                )
                .eq("id", partner_id)
                .single()
                .execute()
            )

            if profile_response.data:
                partner = profile_response.data
                partner["partnership_id"] = relationship["id"]

                partners.append(partner)

        return partners

    except Exception as error:
        st.error(
            f"Could not load accountability partners: {error}"
        )
        return []

#---------------------------------------------------

# Load one user's public MyG profile.

def get_user_profile(user_id):

    try:
        response = (
            supabase.table("profiles")
            .select("*")
            .eq("id", user_id)
            .single()
            .execute()
        )

        return response.data

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. "
                "Please log in again."
            )

        st.error(
            f"Could not load profile: {error}"
        )
        return None

#---------------------------------------------------

# Save changes to the logged-in user's profile.

def update_user_profile(
    user_id,
    username,
    display_name,
    bio
):

    try:
        response = (
            supabase.table("profiles")
            .update({
                "username": username,
                "display_name": display_name,
                "bio": bio,
                "updated_at": datetime.utcnow().isoformat(),
            })
            .eq("id", user_id)
            .execute()
        )

        return bool(response.data)

    except Exception as error:
        if "duplicate key" in str(error).lower():
            st.error(
                "That username is already taken. "
                "Please choose another one."
            )
            return False

        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. "
                "Please log in again."
            )

        st.error(
            f"Could not update profile: {error}"
        )
        return False

#---------------------------------------------------

# Allow the logged-in user to view and edit
# their public MyG identity.

def show_profile_page():

    show_feedback_link("top")

    st.title("👤 My Profile")

    profile = get_user_profile(
        st.session_state.user.id
    )

    if not profile:
        st.error(
            "Your profile could not be loaded."
        )
        return

    st.caption(
        "Your profile is used for accountability "
        "partner invitations and social features."
    )

    username = st.text_input(
        "Username",
        value=profile.get("username") or "",
        max_chars=30,
        placeholder="Example: osaretin"
    )

    st.caption(
        "Your username must be unique. "
        "Other users will eventually use it to find you."
    )

    display_name = st.text_input(
        "Display Name",
        value=profile.get("display_name") or "",
        max_chars=60,
        placeholder="Example: Osaretin"
    )

    bio = st.text_area(
        "Bio",
        value=profile.get("bio") or "",
        max_chars=250,
        placeholder=(
            "Tell your accountability partners "
            "a little about yourself."
        )
    )

    if st.button(
        "Save Profile",
        type="primary"
    ):
        cleaned_username = (
            username.strip().lower()
        )

        if len(cleaned_username) < 3:
            st.warning(
                "Username must contain at least "
                "3 characters."
            )

        elif not re.fullmatch(
            r"[a-z0-9_]+",
            cleaned_username
        ):
            st.warning(
                "Username can only contain lowercase "
                "letters, numbers, and underscores."
            )

        elif not display_name.strip():
            st.warning(
                "Please enter a display name."
            )

        else:
            saved = update_user_profile(
                st.session_state.user.id,
                cleaned_username,
                display_name.strip(),
                bio.strip()
            )

            if saved:
                st.success(
                    "Profile saved successfully."
                )

    show_feedback_link("bottom")

#---------------------------------------------------

# Automatically saves one Progress Tracker step to Supabase.

def autosave_single_step(plan_id, step_number, status, notes):

    if supabase is None or not plan_id:
        return False

    try:
        (
            supabase.table("plan_steps")
            .update({
                "status": status,
                "notes": notes,
                "updated_at": datetime.utcnow().isoformat()
            })
            .eq("plan_id", plan_id)
            .eq("step_number", step_number)
            .execute()
        )

        return True

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. "
                "Please log in again."
            )

        st.session_state.autosave_error = str(error)
        return False


#--------------------------------------------------

#  Reads the current Progress Tracker widget values, 
# updates session state, and saves them to Supabase

def autosave_checkin(step_index, status_key, notes_key):

    if not st.session_state.plan:
        return

    if not st.session_state.current_plan_id:
        return

    steps = st.session_state.plan.get("steps", [])

    if step_index >= len(steps):
        return

    step = steps[step_index]

    status = st.session_state.get(
        status_key,
        step.get("status", "Not Started")
    )

    notes = st.session_state.get(
        notes_key,
        step.get("notes", "")
    )

    # Update the copy in Streamlit memory.
    step["status"] = status
    step["notes"] = notes
    st.session_state.plan["steps"][step_index] = step

    # Permanently save this one step.
    saved = autosave_single_step(
        st.session_state.current_plan_id,
        step.get("step_number", step_index + 1),
        status,
        notes
    )

    if saved:
        st.session_state.autosave_message = "Saved automatically."
        st.session_state.autosave_error = ""
    else:
        st.session_state.autosave_message = ""

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
# Share one of the logged-in user's plans with
# an accepted accountability partner.

def share_plan_with_partner(plan_id, partner_id):

    user_id = st.session_state.user.id

    try:
        response = (
            supabase.table("plan_shares")
            .upsert(
                {
                    "plan_id": plan_id,
                    "owner_id": user_id,
                    "partner_id": partner_id,
                    "can_comment": True,
                    "can_react": True,
                },
                on_conflict="plan_id,partner_id"
            )
            .execute()
        )

        return bool(response.data)

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. "
                "Please log in again."
            )

        st.error(
            f"Could not share plan: {error}"
        )
        return False
#---------------------------------------------------
# Return the accountability partners who currently
# have access to a specific plan.

def get_plan_shares(plan_id):
    """
    Return the accountability partners who currently
    have access to a specific plan.
    """

    try:
        response = (
            supabase.table("plan_shares")
            .select(
                """
                id,
                partner_id,
                can_comment,
                can_react,
                partner:profiles!plan_shares_partner_id_fkey(
                    id,
                    username,
                    display_name
                )
                """
            )
            .eq("plan_id", plan_id)
            .execute()
        )

        return response.data or []

    except Exception as error:
        st.error(
            f"Could not load plan sharing information: {error}"
        )
        return []

#---------------------------------------------------
# Stop sharing a plan with an accountability partner.

def remove_plan_share(share_id):
    """
    Stop sharing a plan with an accountability partner.
    """

    try:
        response = (
            supabase.table("plan_shares")
            .delete()
            .eq("id", share_id)
            .execute()
        )

        return bool(response.data)

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. "
                "Please log in again."
            )

        st.error(
            f"Could not stop sharing plan: {error}"
        )
        return False

#---------------------------------------------------
# load plans shared with the logged-in user

def get_plans_shared_with_me():
    """
    Return plans explicitly shared with the logged-in user.

    This deliberately loads the share, plan, and owner profile
    separately so the logic is easier to understand and debug.
    """

    user_id = st.session_state.user.id

    try:
        # 1. Find shares where the logged-in user is the partner.
        shares_response = (
            supabase.table("plan_shares")
            .select("*")
            .eq("partner_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        shares = shares_response.data or []

        shared_plans = []

        # 2. Load the corresponding plan and owner for each share.
        for share in shares:

            plan_response = (
                supabase.table("plans")
                .select(
                    "id, goal_summary, overall_deadline, "
                    "likely_obstacle, today_next_action, created_at"
                )
                .eq("id", share["plan_id"])
                .execute()
            )

            if not plan_response.data:
                continue

            owner_response = (
                supabase.table("profiles")
                .select(
                    "id, username, display_name"
                )
                .eq("id", share["owner_id"])
                .execute()
            )

            owner = (
                owner_response.data[0]
                if owner_response.data
                else {}
            )

            plan = plan_response.data[0]

            # Preserve the structure expected by
            # show_shared_plans_page().
            share["plan"] = plan
            share["owner"] = owner

            shared_plans.append(share)

        return shared_plans

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. "
                "Please log in again."
            )

        st.error(
            f"Could not load shared plans: {error}"
        )
        return []

#---------------------------------------------------
# load one shared plan with its steps

def load_shared_plan(plan_id):
    """
    Load a plan and its steps when the current user
    has permission through plan_shares.
    """

    try:
        plan_response = (
            supabase.table("plans")
            .select("*")
            .eq("id", plan_id)
            .single()
            .execute()
        )

        steps_response = (
            supabase.table("plan_steps")
            .select("*")
            .eq("plan_id", plan_id)
            .order("step_number")
            .execute()
        )

        if not plan_response.data:
            return None

        plan = plan_response.data
        plan["steps"] = steps_response.data or []

        return plan

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. "
                "Please log in again."
            )

        st.error(
            f"Could not load shared plan: {error}"
        )
        return None

def add_step_comment(plan_id, step_id, comment_text):
    """
    Add a comment to a specific step of a shared plan.
    """

    comment_text = comment_text.strip()

    if not comment_text:
        return False

    if len(comment_text) > 1000:
        st.warning("Comments cannot exceed 1,000 characters.")
        return False

    try:
        response = (
            supabase.table("step_comments")
            .insert({
                "plan_id": plan_id,
                "step_id": step_id,
                "author_id": st.session_state.user.id,
                "comment_text": comment_text,
            })
            .execute()
        )

        return bool(response.data)

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. Please log in again."
            )

        st.error(f"Could not post comment: {error}")
        return False


def get_step_comments(step_id):
    """
    Return comments for a particular plan step,
    oldest comment first.
    """

    try:
        response = (
            supabase.table("step_comments")
            .select(
                """
                id,
                step_id,
                plan_id,
                author_id,
                comment_text,
                created_at,
                author:profiles!step_comments_author_id_fkey(
                    id,
                    username,
                    display_name
                )
                """
            )
            .eq("step_id", step_id)
            .order("created_at")
            .execute()
        )

        return response.data or []

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. Please log in again."
            )

        st.error(f"Could not load comments: {error}")
        return []


def delete_step_comment(comment_id):
    """
    Delete a comment written by the logged-in user.
    RLS prevents users from deleting someone else's comment.
    """

    try:
        response = (
            supabase.table("step_comments")
            .delete()
            .eq("id", comment_id)
            .eq(
                "author_id",
                st.session_state.user.id
            )
            .execute()
        )

        return bool(response.data)

    except Exception as error:
        if is_auth_session_error(error):
            return_user_to_login(
                "Your login session expired. Please log in again."
            )

        st.error(f"Could not delete comment: {error}")
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
        same_site="lax",
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
        same_site="lax",
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

    raw_cookie = cookie_manager.get(AUTH_COOKIE_NAME)

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
# Partner View Page

def show_shared_plans_page():
    """
    Shows plans that other users have explicitly shared
    with the logged-in accountability partner.
    """

    show_feedback_link("top")

    st.title("👀 Plans Shared With Me")

    shared_plans = get_plans_shared_with_me()

    if not shared_plans:
        st.info(
            "No accountability plans have been shared with you yet."
        )
        show_feedback_link("bottom")
        return

    for share in shared_plans:
        owner = share.get("owner") or {}
        plan = share.get("plan") or {}

        owner_name = (
            owner.get("display_name")
            or owner.get("username")
            or "MyG User"
        )

        st.subheader(
            plan.get("goal_summary", "Untitled Plan")
        )

        st.caption(
            f"Shared by {owner_name}"
        )

        st.write(
            f"**Deadline:** "
            f"{plan.get('overall_deadline', 'No deadline')}"
        )

        st.write(
            f"**Next Action:** "
            f"{plan.get('today_next_action', 'No next action')}"
        )

        if st.button(
            "Open Shared Plan",
            key=f"open_shared_{share['id']}"
        ):
            loaded_plan = load_shared_plan(
                plan["id"]
            )

            if loaded_plan:
                st.session_state.shared_plan = loaded_plan
                st.session_state.shared_plan_share = share

                set_current_page("shared_plan_view")
                st.rerun()

        st.markdown("---")

    show_feedback_link("bottom")

#------------------------------------------
# Read-Only View of a Shared Plan

def show_shared_plan_view():
    """
    Read-only view of a plan shared with the current partner.
    """

    show_feedback_link("top")

    plan = st.session_state.get("shared_plan")
    share = st.session_state.get("shared_plan_share")

    if not plan or not share:
        st.warning(
            "No shared plan is currently selected."
        )

        if st.button("Back to Shared Plans"):
            set_current_page("shared_plans")
            st.rerun()

        return

    st.title("🤝 Shared Accountability Plan")

    st.subheader(plan.get("goal_summary", "Untitled Plan"))

    st.write(
        f"**Deadline:** "
        f"{plan.get('overall_deadline', 'No deadline')}"
    )

    st.write(
        f"**Likely Obstacle:** "
        f"{plan.get('likely_obstacle', '')}"
    )

    st.write(
        f"**Today's Next Action:** "
        f"{plan.get('today_next_action', '')}"
    )

    st.markdown("---")
    st.subheader("Plan Steps")

    for step in plan.get("steps", []):
        step_id = step.get("id")

        st.write(
            f"### Step {step.get('step_number')}: "
            f"{step.get('step_name', 'Untitled Step')}"
        )

        st.write(
            f"**Description:** "
            f"{step.get('description', '')}"
        )

        st.write(
            f"**Time Block:** "
            f"{step.get('time_block', '')}"
        )

        st.write(
            f"**Deadline:** "
            f"{step.get('deadline', '')}"
        )

        st.write(
            f"**Status:** "
            f"{step.get('status', 'Not Started')}"
        )

        if step.get("notes"):
            st.write(
                f"**Latest Check-In:** "
                f"{step.get('notes')}"
            )

        # ---------------------------------
        # Accountability comments
        # ---------------------------------
        st.markdown("#### 💬 Accountability")

        comments = get_step_comments(step_id)

        if not comments:
            st.caption("No comments yet.")

        for comment in comments:
            author = comment.get("author") or {}

            author_name = (
                author.get("display_name")
                or author.get("username")
                or "MyG User"
            )

            st.write(f"**{author_name}**")
            st.write(comment.get("comment_text", ""))

            # Only show Delete for comments written
            # by the currently logged-in user.
            if (
                comment.get("author_id")
                == st.session_state.user.id
            ):
                if st.button(
                    "Delete",
                    key=f"delete_comment_{comment['id']}"
                ):
                    deleted = delete_step_comment(
                        comment["id"]
                    )

                    if deleted:
                        st.rerun()

        # ---------------------------------
        # New comment form
        # ---------------------------------
        if share.get("can_comment", False):

            with st.form(
                f"comment_form_{step_id}",
                clear_on_submit=True
            ):
                comment_text = st.text_area(
                    "Add a comment",
                    placeholder=(
                        "Write some encouragement, advice, "
                        "or accountability feedback..."
                    ),
                    max_chars=1000
                )

                comment_submitted = st.form_submit_button(
                    "Post Comment"
                )

            if comment_submitted:
                if not comment_text.strip():
                    st.warning(
                        "Please write a comment before posting."
                    )

                else:
                    posted = add_step_comment(
                        plan["id"],
                        step_id,
                        comment_text
                    )

                    if posted:
                        st.success("Comment posted.")
                        st.rerun()

        else:
            st.caption(
                "Comments are disabled for this shared plan."
            )

        st.markdown("---")

    if st.button("Back to Shared Plans"):
        set_current_page("shared_plans")
        st.rerun()

    show_feedback_link("bottom")

#------------------------------------------

# Social hub for accountability partnerships.

def show_partners_page():

    show_feedback_link("top")

    st.title("🤝 Accountability Partners")

    st.write(
        "Connect with someone you trust to help "
        "support and encourage your progress."
    )


    # ---------------------------------
    # Find another MyG user
    # ---------------------------------
    st.subheader("Find a Partner")

    username_search = st.text_input(
        "Search by username",
        placeholder="Example: osaretin"
    )

    if st.button("Search User"):
        if not username_search.strip():
            st.warning(
                "Enter a username first."
            )

        else:
            profile = search_profile_by_username(
                username_search
            )

            if not profile:
                st.info(
                    "No user was found with that username."
                )

            elif (
                profile["id"]
                == st.session_state.user.id
            ):
                st.info(
                    "That's your own profile."
                )

            else:
                st.session_state[
                    "partner_search_result"
                ] = profile


    profile = st.session_state.get(
        "partner_search_result"
    )

    if profile:
        st.markdown("---")

        st.write(
            f"### {profile.get('display_name') or profile.get('username')}"
        )

        st.write(
            f"@{profile.get('username', '')}"
        )

        if profile.get("bio"):
            st.write(profile["bio"])

        if st.button(
            "Send Accountability Request",
            type="primary"
        ):
            sent = send_partnership_request(
                profile["id"]
            )

            if sent:
                st.success(
                    "Accountability partner request sent!"
                )

                st.session_state.pop(
                    "partner_search_result",
                    None
                )


    # ---------------------------------
    # Incoming requests
    # ---------------------------------
    st.markdown("---")
    st.subheader("Partner Requests")

    requests = get_incoming_partner_requests()

    if not requests:
        st.caption(
            "You have no pending partner requests."
        )

    for request in requests:
        requester = request.get(
            "requester",
            {}
        ) or {}

        st.write(
            f"**{requester.get('display_name') or requester.get('username', 'MyG User')}**"
        )

        st.caption(
            f"@{requester.get('username', '')}"
        )

        col_accept, col_decline = st.columns(2)

        with col_accept:
            if st.button(
                "Accept",
                key=f"accept_{request['id']}",
                type="primary"
            ):
                if respond_to_partner_request(
                    request["id"],
                    "accepted"
                ):
                    st.success(
                        "Accountability partnership accepted!"
                    )
                    st.rerun()

        with col_decline:
            if st.button(
                "Decline",
                key=f"decline_{request['id']}"
            ):
                if respond_to_partner_request(
                    request["id"],
                    "declined"
                ):
                    st.rerun()


    # ---------------------------------
    # Existing partners
    # ---------------------------------
    st.markdown("---")
    st.subheader("My Accountability Partners")

    partners = get_accountability_partners()

    if not partners:
        st.caption(
            "You do not have an accountability partner yet."
        )

    for partner in partners:
        st.write(
            f"### {partner.get('display_name') or partner.get('username')}"
        )

        st.caption(
            f"@{partner.get('username', '')}"
        )

        if partner.get("bio"):
            st.write(partner["bio"])

        st.markdown("---")

    show_feedback_link("bottom")

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

            # ---------------------------------
            # Accountability plan sharing
            # ---------------------------------
            st.markdown("#### 🤝 Accountability")

            partners = get_accountability_partners()

            if not partners:
                st.caption(
                    "Add an accountability partner before sharing this plan."
                )

            else:
                partner_options = {
                    (
                        partner.get("display_name")
                        or partner.get("username")
                        or "MyG User"
                    ): partner
                    for partner in partners
                }

                selected_partner_name = st.selectbox(
                    "Share this plan with",
                    list(partner_options.keys()),
                    key=f"share_partner_{plan['id']}"
                )

                selected_partner = partner_options[selected_partner_name]

                if st.button(
                    "Share Plan",
                    key=f"share_plan_{plan['id']}"
                ):
                    shared = share_plan_with_partner(
                        plan["id"],
                        selected_partner["id"]
                    )

                    if shared:
                        st.success(
                            f"Plan shared with {selected_partner_name}."
                        )
                        

            current_shares = get_plan_shares(plan["id"])

            if current_shares:
                st.caption("Currently shared with:")

                for share in current_shares:
                    partner = share.get("partner") or {}

                    partner_name = (
                        partner.get("display_name")
                        or partner.get("username")
                        or "MyG User"
                    )

                    col_name, col_remove = st.columns([3, 1])

                    with col_name:
                        st.write(f"🤝 {partner_name}")

                    with col_remove:
                        if st.button(
                            "Remove",
                            key=f"remove_share_{share['id']}"
                        ):
                            removed = remove_plan_share(
                                share["id"]
                            )

                            if removed:
                                st.rerun()

            st.markdown("---")



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
    "profile",
    "partners",
    "shared_plans",
    "shared_plan_view",
    "Shared With Me",
}

PAGE_TO_SIDEBAR = {
    "my_plans": "My Plans",
    "goal_setup": "Create New Plan",
    "daily_checkin": "Progress Tracker",
    "partners": "Accountability Partners",
    "shared_plans": "Shared With Me",
    "shared_plan_view": "Shared With Me",
    "profile": "Profile",
}

def set_current_page(page_name):
    """
    Update the app's current page and browser URL.
    """

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

# CookieManager may need one Streamlit render cycle before browser
# cookies become available through cookie_manager.get().
cookie_manager = stx.CookieManager(
    key="myg_cookie_manager"
)

# -----------------------------------------
# Restore persistent Supabase authentication
# -----------------------------------------
if (
    st.session_state.user is None
    and not st.session_state.get("logout_in_progress", False)
):
    restored = restore_login_from_cookie()

    if restored:
        # Authentication was restored successfully.
        st.session_state.auth_restore_attempts = 0

    elif st.session_state.auth_restore_attempts < 1:
        # On the first run, CookieManager may not have loaded
        # the browser cookies yet. Do not show the login page.
        st.session_state.auth_restore_attempts += 1

        st.info("Restoring your session...")
        st.stop()


# -----------------------------------------
# Complete postponed cookie operations
# -----------------------------------------
if st.session_state.pop(
    "pending_auth_cookie_delete",
    False
):
    delete_auth_cookie()

pending_cookie_session = st.session_state.pop(
    "pending_auth_cookie_session",
    None
)

if pending_cookie_session is not None:
    save_auth_cookie(pending_cookie_session)


# -----------------------------------------
# Show login only after restoration was tried
# -----------------------------------------
if st.session_state.user is None:
    show_login_page()
    st.stop()


# Keep the browser cookie synchronized if Supabase
# rotates the access or refresh tokens.
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


# -----------------------------------------
# Sidebar navigation
# -----------------------------------------

PAGE_LABELS = {
    "My Plans": "my_plans",
    "Create New Plan": "goal_setup",
    "Progress Tracker": "daily_checkin",
    "Accountability Partners": "partners",
    "Shared With Me": "shared_plans",
    "Profile": "profile",
}

PAGE_NAMES = {
    page_name: label
    for label, page_name in PAGE_LABELS.items()
}

# shared_plan_view is a child page of Shared With Me.
sidebar_page = st.session_state.page

if sidebar_page == "shared_plan_view":
    sidebar_page = "shared_plans"

current_label = PAGE_NAMES.get(
    sidebar_page,
    "My Plans"
)

selected_page = st.sidebar.radio(
    "Go to",
    list(PAGE_LABELS.keys()),
    index=list(PAGE_LABELS.keys()).index(current_label)
)

selected_page_name = PAGE_LABELS[selected_page]

if selected_page_name != sidebar_page:
    set_current_page(selected_page_name)
    st.rerun()
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

if st.session_state.page == "partners":
    show_partners_page()
    st.stop()

if st.session_state.page == "profile":
    show_profile_page()
    st.stop()

if st.session_state.page == "shared_plans":
    show_shared_plans_page()
    st.stop()

if st.session_state.page == "shared_plan_view":
    show_shared_plan_view()
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
    key="goal",
    placeholder=(
        "Example: I want to lose weight, build my business, "
        "study better, or save money."
    ),
    max_chars=None
)

if st.button("Generate SMART Suggestions"):
    if not goal.strip():
        st.warning("Please enter a goal first.")
    else:
        # st.session_state.goal = goal.strip()

        with st.spinner("Generating SMART suggestions..."):
            try:
                st.session_state.smart_suggestions = generate_smart_suggestions(goal.strip())
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