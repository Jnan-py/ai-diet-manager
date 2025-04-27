import os
import json
import sqlite3
import hashlib
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from phi.agent import Agent
from phi.tools.website import WebsiteTools
from phi.tools.wikipedia import WikipediaTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.serpapi_tools import SerpApiTools
from phi.model.google import Gemini
from PIL import Image
import re
from pydantic import BaseModel, Field
from typing import List, Dict, Any


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
if not GOOGLE_API_KEY or not SERPAPI_API_KEY:
    st.error("Please set GOOGLE_API_KEY and SERPAPI_API_KEY in your .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)


DB_PATH = 'data.db'
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE,
    password_hash TEXT
)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS diets (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    name TEXT,
    preferences TEXT,
    responses TEXT,
    plan TEXT,
    advice TEXT,
    created_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    diet_id INTEGER,
    role TEXT,
    message TEXT,
    timestamp TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id),
    FOREIGN KEY(diet_id) REFERENCES diets(id)
)''')
conn.commit()


food_analysis_agent = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=(
        "You are a nutrition expert. Given an image of a dish, identify the specific dish, "
        "list key ingredients (e.g., rice, pulses, spices), and compute nutritional values per gram: "
        "calories, fat, protein, carbs, and key vitamins/minerals. Return STRICT JSON: "
        "{'food_type': string, 'ingredients': [string], 'nutrition_per_g': {"
        "'calories': float, 'fat_g': float, 'protein_g': float, 'carbs_g': float, "
        "'vitamins': {string: float}}}"
    )
)

def create_agent(instruction: str, response_model=None):
    return Agent(
        name="Agent",
        model=Gemini(id="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
        tools=[WebsiteTools(), WikipediaTools(), DuckDuckGo(), SerpApiTools(api_key=SERPAPI_API_KEY)],
        instructions=instruction,
        response_model=response_model,
        structured_output=bool(response_model)
    )

class WeeklyPlan(BaseModel):
    plan: Dict[str, Dict[str, Any]] = Field(..., description="Mapping of day to meals and details")
    citations: List[str] = Field(..., description="List of citation URLs")

class DailyAdvice(BaseModel):
    daily_calories: float = Field(..., description="Recommended daily calorie intake")
    tips: List[str] = Field(..., description="Daily intake tips")
    citations: List[str] = Field(..., description="List of citation URLs")


diet_agent = Agent(
    name="DietPlanner",
    model=Gemini(id="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
    tools=[WebsiteTools(), WikipediaTools(), DuckDuckGo(), SerpApiTools(api_key=SERPAPI_API_KEY)],
    instructions=(
        "Using web sources and your own knowledge, generate a weekly meal plan based on the user's preferences. "
        "Return STRICT JSON: {'plan': { 'Monday': {'Breakfast': '...', 'Lunch': '...', ...}, ... }, 'citations': [url,...]}"
    ),
    response_model=WeeklyPlan,
    structured_output=True
)

calorie_agent = Agent(
    name="CalorieAdvisor",
    model=Gemini(id="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
    tools=[WebsiteTools(), WikipediaTools(), DuckDuckGo(), SerpApiTools(api_key=SERPAPI_API_KEY)],
    instructions=(
        "Using web sources and the user profile, recommend daily calorie targets and intake tips. "
        "Return STRICT JSON: {'daily_calories': float, 'tips': [string,...], 'citations': [url,...]}"
    ),
    response_model=DailyAdvice,
    structured_output=True
)

alignment_agent = create_agent(
    "Given the user's diet preferences, weekly meal plan, and a detected dish (ingredients and nutrition), determine whether this food is essential (aligned) or non-essential to the diet. Provide a reason and suggest alternative foods that better fit the diet. Return STRICT JSON: { 'essential': bool, 'reason': string, 'alternatives': [string] }"
)


chat_agent = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=(
        "You are a diet editing assistant. Given the user's current diet preferences, weekly meal plan, and calorie advice, "
        "help the user make modifications."
        "Make sure to include the user's preferences and any specific requests in your response."
        "Make sure that the modifications align with the user's preferences and dietary goals."
        "If the user asks for a specific food or meal, provide alternatives that fit their diet."
        "Respond with STRICT JSON: {'preferences': string, 'plan': {...}, 'advice': {'daily_calories': float, 'tips': [string,...]}}"
    )
)

bot_agent = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=(
        "You are a diet assistant. Given the user's preferences and dietary goals, "
        "provide personalized diet advice. "
        "Make sure that the advice aligns with the user's preferences and dietary goals."
        "Please answer in a conversational manner."        
    )
)


def parse_gemini_response(text: str) -> Any:
    clean = re.sub(r'```json|```', '', text)
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None

def generate_question(page: int, prefs: str, previous_responses: List[Dict[str,Any]]) -> Dict[str,Any]:    
    if page == 1:
        prompt = (
            f"User Preferences: {prefs}\n"
            "Generate the first diet customization question to gather essential user info "
            "(medical issues, allergies, dietary habits, etc.). "
            "Return STRICT JSON: {'id': 1, 'question': string}."
        )
    else:
        prompt = (
            f"User Preferences: {prefs}\n"
            f"Previous Q&A: {json.dumps(previous_responses)}\n"
            f"Generate diet customization question #{page} based on the above responses. "
            "Return STRICT JSON: {'id': <id>, 'question': string}."
        )
    agent = Agent(
        name="QuestionGenerator",
        model=Gemini(id="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
        tools=[WebsiteTools(), WikipediaTools(), DuckDuckGo(), SerpApiTools(api_key=SERPAPI_API_KEY)],
        instructions="Generate a single follow-up diet customization question as JSON."
    )
    resp = agent.run(prompt)
    qdata = parse_gemini_response(resp.content)
    if not isinstance(qdata, dict) or 'id' not in qdata or 'question' not in qdata:
        st.error("Failed to generate a valid question. Please try again.")
        return {'id': page, 'question': 'Could not generate question, please provide details.'}
    return qdata


def save_chat_message(user_id: int, diet_id: int, role: str, message: str):
    cursor.execute(
        "INSERT INTO chat_history (user_id, diet_id, role, message, timestamp) VALUES (?,?,?,?,?)",
        (user_id, diet_id, role, message, datetime.utcnow().isoformat())
    )
    conn.commit()


def fetch_chat_history(user_id: int, diet_id: int) -> List[Dict[str, str]]:
    cursor.execute(
        "SELECT role, message FROM chat_history WHERE user_id=? AND diet_id=? ORDER BY timestamp ASC",
        (user_id, diet_id)
    )
    return [{'role': row[0], 'content': row[1]} for row in cursor.fetchall()]

hash_password = lambda pw: hashlib.sha256(pw.encode()).hexdigest()
def create_user(username: str, password: str) -> bool:
    try:
        cursor.execute("INSERT INTO users (username,password_hash) VALUES (?,?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username: str, password: str) -> bool:
    cursor.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    row = cursor.fetchone()
    return bool(row and row[0] == hash_password(password))

def get_user_id(username: str) -> int:
    cursor.execute("SELECT id FROM users WHERE username=?", (username,))
    row = cursor.fetchone()
    return row[0] if row else None

def save_diet(user_id: int, name: str, prefs: str, responses: List[Any], plan: Dict[str,Any], advice: Dict[str,Any]) -> int:
    cursor.execute(
        "INSERT INTO diets (user_id,name,preferences,responses,plan,advice,created_at) VALUES (?,?,?,?,?,?,?)",
        (user_id, name, prefs, json.dumps(responses), json.dumps(plan), json.dumps(advice), datetime.utcnow().isoformat())
    )
    conn.commit()
    return cursor.lastrowid

def fetch_diets(user_id: int) -> List[tuple]:
    cursor.execute("SELECT id,name FROM diets WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    return cursor.fetchall()

def fetch_diet(diet_id: int) -> dict:
    cursor.execute("SELECT name,preferences,responses,plan,advice FROM diets WHERE id=?", (diet_id,))
    row = cursor.fetchone()
    if not row:
        return {}
    name, prefs, responses, plan, advice = row
    return { 'name': name, 'preferences': prefs,
             'responses': json.loads(responses), 'plan': json.loads(plan), 'advice': json.loads(advice) }


st.set_page_config(page_title="AI Diet Manager", layout="wide")
if 'logged_in' not in st.session_state:
    st.session_state.update({
        'logged_in': False,
        'username': '',
        'new_diet': False,
        'selected_diet': None,
        'flow': {'step':0,'name':'','prefs':'','responses':[]},
        'edit_mode': None
    })

st.title("AI Diet Manager")

if not st.session_state.logged_in:    
    with st.sidebar:
        st.title("Login / Signup")
        select = st.selectbox("Select", ["Login", "Sign Up"])

    if select == "Login": 
        st.header("Login") 
        with st.form(key="login_form"):
            login_user = st.text_input("Username")
            login_pw = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if verify_user(login_user, login_pw):
                    st.session_state.logged_in = True
                    st.session_state.username = login_user
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

    else:
        st.header("Sign Up")
        with st.form(key="signup_form"):
            signup_user = st.text_input("New Username")
            signup_pw = st.text_input("New Password", type="password")
            if st.form_submit_button("Sign Up"):
                if create_user(signup_user, signup_pw):
                    st.success("Account created! Please log in.")
                else:
                    st.error("Username already exists.")
    st.stop()


uid = get_user_id(st.session_state.username)
st.sidebar.title(f"👤 Welcome {st.session_state.username} !!")
if st.sidebar.button("Create New Diet"):
    st.session_state.new_diet = True
    st.session_state.selected_diet = None
    st.session_state.edit_mode = None
    st.session_state.flow = {'step':0,'name':'','prefs':'','responses':[]}
    st.rerun()

diets = fetch_diets(uid)
if diets:
    names = [name for (_id,name) in diets]
    sel_name = st.sidebar.selectbox("Your Diet Plans", names)
    if st.sidebar.button("Load Diet"):
        for did,name in diets:
            if name == sel_name:
                st.session_state.selected_diet = did
                st.session_state.new_diet = False
                st.session_state.edit_mode = None
                st.rerun()

if st.session_state.new_diet:
    st.subheader("Diet Plan Generation")
    flow = st.session_state.flow
    
    if flow['step']==0:        
        st.markdown("Please provide a name for your diet plan.")
        with st.form(key="form0"):
            nm = st.text_input("Diet Plan Name")
            if st.form_submit_button("Next") and nm.strip():
                flow['name']=nm.strip(); flow['step']=1; st.rerun()
    
    elif flow['step']==1:
        st.markdown("Please enter your preferences / Goals.")
        with st.form(key="form1"):
            pf = st.text_input("Preferences/Goals")
            if st.form_submit_button("Next") and pf.strip():
                flow['prefs']=pf.strip(); flow['step']=2; st.rerun()
    
    elif 2 <= flow['step'] <= 11:
        st.markdown("#### Questionnaire")
        st.markdown("Please answer the following questions to help us customize your diet plan.")
        st.markdown("---")
        page = flow['step'] - 1        

        with st.spinner("Generating question..."):
            if 'current_question' not in flow:
                flow['current_question'] = generate_question(page, flow['prefs'], flow['responses'])
            q = flow['current_question']

            st.markdown(f"**Question {q['id']}:** {q['question']}")
            ans = st.text_input("Your answer:", key=f"answer_{q['id']}")
            if st.button("Submit Answer"):
                if not ans.strip():
                    st.warning("Please enter an answer before proceeding.")
                else:
                    flow['responses'].append({
                        'id': q['id'],
                        'question': q['question'],
                        'answer': ans.strip()
                    })
                    del flow['current_question']
                    flow['step'] += 1
                    st.rerun()
    
    else:
        fp = f"Preferences: {flow['prefs']}. Q&A: {json.dumps(flow['responses'])}"
        wk = diet_agent.run(fp).content
        adv= calorie_agent.run(fp).content
        st.subheader("Weekly Meal Plan")
        for d,ms in wk.plan.items():
            st.markdown(f"### {d}")
            for m,dsc in ms.items(): st.markdown(f"- **{m}:** {dsc}")
        st.subheader("Calorie Advice")
        st.metric("Daily Calories", adv.daily_calories)
        st.markdown("### Tips")
        for t in adv.tips: st.markdown(f"- {t}")
        
        with st.form(key="form_save"):
            if st.form_submit_button("Save Diet Plan"):
                new_id=save_diet(uid,flow['name'],flow['prefs'],flow['responses'],wk.plan,adv.dict())
                st.session_state.new_diet=False
                st.session_state.selected_diet=new_id
                st.rerun()

elif st.session_state.selected_diet:
    data = fetch_diet(st.session_state.selected_diet)
    t1, t2, t3 = st.tabs(["Diet Plan","Food Analysis", "Chatbot Assistant"])

    with t1:
        st.subheader(data['name'])
        st.markdown(f"**Preferences:** {data['preferences']}")
        st.markdown("### Weekly Plan")
        for d, meals in data['plan'].items():
            st.markdown(f"#### {d}")
            for meal, desc in meals.items():
                st.markdown(f"- **{meal}:** {desc}")
        st.subheader("Calorie Advice")
        st.metric("Daily Calories", data['advice']['daily_calories'])
        st.markdown("### Tips")
        for tip in data['advice']['tips']:
            st.markdown(f"- {tip}")
        
        col1, col2 = st.columns(2)
        if col1.button("Edit with Preferences & Questionnaire"):
            st.session_state.edit_mode = 'form'
            st.session_state.new_diet = True
            st.session_state.selected_diet = None
            st.session_state.flow = {
                'step': 1,
                'name': data['name'],
                'prefs': data['preferences'],
                'responses': []
            }
            st.rerun()

        if col2.button("Edit using Chatbot"):
            st.session_state.edit_mode = 'chatbot'
            st.session_state.chat_history = []
            initial_msg = (
                f"Current Diet Plan:\n{json.dumps(data['plan'], indent=2)}\n\n"
                f"Preferences: {data['preferences']}\n"
                f"Calorie Advice: {data['advice']['daily_calories']} calories/day\n"
                f"Tips: {'; '.join(data['advice']['tips'])}"
            )
            st.session_state.chat_history.append({'role': 'assistant', 'content': initial_msg})
            st.session_state.chat_history.append({'role': 'assistant', 'content': "Please provide me the plans and preferences useful to modify your diet plan."})
            st.rerun()

        if st.session_state.get('edit_mode') == 'chatbot':
            st.subheader("Chatbot-based Diet Editing")            

            for msg in st.session_state.chat_history[1:]:
                with st.chat_message(msg['role']):
                    if msg['role'] == 'user':
                        st.markdown(msg['content'])
                    else:                        
                        parsed = parse_gemini_response(msg['content'])
                        if parsed and 'plan' in parsed and 'advice' in parsed and 'preferences' in parsed:
                            st.markdown("### Updated Diet Plan")
                            st.markdown(f"**Preferences:** {parsed['preferences']}")
                            for d, meals in parsed['plan'].items():
                                st.markdown(f"#### {d}")
                                for meal, desc in meals.items():
                                    st.markdown(f"- **{meal}:** {desc}")
                            st.subheader("Updated Calorie Advice")
                            st.metric("Daily Calories", parsed['advice']['daily_calories'])
                            st.markdown("### Updated Tips")
                            for tip in parsed['advice']['tips']:
                                st.markdown(f"- {tip}")
                        else:
                            st.markdown(msg['content'])

            if user_input := st.chat_input("Ask me to modify your diet plan"):
                st.session_state.chat_history.append({'role':'user','content':user_input})
                with st.spinner("Modifying Diet..."):
                    history_text = ""
                    for m in st.session_state.chat_history:
                        history_text += f"{m['role'].capitalize()}: {m['content']}\n"
                    response = chat_agent.generate_content(history_text)
                    st.session_state.chat_history.append({'role':'assistant','content':response.text})     

                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    parsed = parse_gemini_response(response.text)
                    if parsed and 'plan' in parsed and 'advice' in parsed and 'preferences' in parsed:
                        st.markdown("### Updated Diet Plan")
                        st.markdown(f"**Preferences:** {parsed['preferences']}")
                        for d, meals in parsed['plan'].items():
                            st.markdown(f"#### {d}")
                            for meal, desc in meals.items():
                                st.markdown(f"- **{meal}:** {desc}")
                        st.subheader("Updated Calorie Advice")
                        st.metric("Daily Calories", parsed['advice']['daily_calories'])
                        st.markdown("### Updated Tips")
                        for tip in parsed['advice']['tips']:
                            st.markdown(f"- {tip}")

            if st.button("Save the Plan"):
                last_content = st.session_state.chat_history[-1]['content']
                parsed = parse_gemini_response(last_content)
                if parsed and 'plan' in parsed and 'advice' in parsed and 'preferences' in parsed:
                    cursor.execute(
                        "UPDATE diets SET preferences=?, plan=?, advice=?, created_at=? WHERE id=?",
                        (
                            json.dumps(parsed['preferences']),  
                            json.dumps(parsed['plan']),
                            json.dumps(parsed['advice']),
                            datetime.utcnow().isoformat(),
                            st.session_state.selected_diet
                        )
                    )
                    conn.commit()
                    st.success("Plan updated successfully!")
                    st.session_state.edit_mode = None
                    st.rerun()
                else:
                    st.error("Failed to parse the updated plan. Ensure the chatbot response contains valid JSON.")
            
            st.stop()


    with t2:
        st.header("Food Analysis & Alignment")
        uploaded = st.file_uploader("Upload food image", type=["png","jpg","jpeg"])
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, width=300, caption="Uploaded Food Image")
            if st.button("Analyze Food"):

                with st.spinner("Analyzing food contents..."):
                    r = food_analysis_agent.generate_content(img)
                    res = parse_gemini_response(r.text)
                    if res:
                        st.subheader(res['food_type'])
                        st.markdown("**Ingredients:**")
                        for ing in res['ingredients']:
                            st.markdown(f"- {ing}")
                        nutr = res['nutrition_per_g']
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Calories", nutr.get('calories', 0))
                        c2.metric("Fat (g)", nutr.get('fat_g', 0))
                        c3.metric("Protein (g)", nutr.get('protein_g', 0))
                        c4.metric("Carbs (g)", nutr.get('carbs_g', 0))
                        if nutr.get('vitamins'):
                            st.subheader("Vitamins & Minerals")
                            for k, v in nutr['vitamins'].items():
                                st.write(f"{k}: {v}")
                        alignment_input = {
                            'preferences': data['preferences'],
                            'weekly_plan': data['plan'],
                            'detected_food': res
                        }
                        align_resp = alignment_agent.run(json.dumps(alignment_input))
                        align = parse_gemini_response(align_resp.content)

                    with st.spinner("Analyzing food alignment with diet..."):
                        if align:
                            st.markdown("---")
                            st.subheader("Diet Alignment")
                            st.write(f"**Aligned with diet?** {'Yes' if align['essential'] else 'No'}")
                            st.write(f"**Reason:** {align.get('reason','')}" )
                            st.markdown("---")
                            det_mod = genai.GenerativeModel(
                                model_name="gemini-2.0-flash",
                                system_instruction=(
                                    "You are a nutrition expert, given a food type and its ingredients, along with a diet plan, "
                                    "determine if the food is aligned with the diet. "
                                    "The alignment Yes/No is already provided. "
                                    "Provide a detailed reason for the alignment status. "
                                    "Provide alternatives and the description too"
                                )
                            )

                            det_resp = det_mod.generate_content(json.dumps(alignment_input) + " " + json.dumps(align))
                        
                            st.markdown("**Detailed Analysis:**")
                            st.markdown(det_resp.text)

                        else:
                            st.error("Analysis failed.")

    with t3:
        st.header("Chatbot Assistant")
        diet_id = st.session_state.selected_diet
        
        if st.session_state.get('last_diet_id') != diet_id:
            st.session_state.last_diet_id = diet_id
            if 'chat_history_bot' in st.session_state:
                del st.session_state['chat_history_bot']

        if 'chat_history_bot' not in st.session_state:
            stored = fetch_chat_history(uid, diet_id)
            if stored:
                st.session_state.chat_history_bot = stored.copy()
            else:
                initial_msg = (
                    f"Current Diet Plan:\n{json.dumps(data['plan'], indent=2)}\n\n"
                    f"Preferences: {data['preferences']}\n"
                    f"Calorie Advice: {data['advice']['daily_calories']} calories/day\n"
                    f"Tips: {'; '.join(data['advice']['tips'])}"
                )
                st.session_state.chat_history_bot = [
                    {'role': 'assistant', 'content': initial_msg},
                    {'role': 'assistant', 'content': "Hello! How can I assist you with your diet plan?"}
                ]
                for msg in st.session_state.chat_history_bot:
                    save_chat_message(uid, diet_id, msg['role'], msg['content'])
        
        for msg in st.session_state.chat_history_bot[1:]:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

        if user_input := st.chat_input("Ask me anything about your diet plan"):
            st.session_state.chat_history_bot.append({'role': 'user', 'content': user_input})
            save_chat_message(uid, diet_id, 'user', user_input)

            with st.spinner("Generating response..."):
                history_text = "".join([f"{m['role'].capitalize()}: {m['content']}\n" for m in st.session_state.chat_history_bot])
                response = bot_agent.generate_content(history_text)
                reply = response.text

                st.session_state.chat_history_bot.append({'role': 'assistant', 'content': reply})
                save_chat_message(uid, diet_id, 'assistant', reply)

            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                st.markdown(reply)
else:
    st.info("Use the sidebar to create or select a diet plan.")