import streamlit as st
import os
import bcrypt
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "")

USER_DB = "users.json"

def get_password_hash(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed_password):
    try:
        if len(hashed_password) == 64:
            return hashlib.sha256(password.encode()).hexdigest() == hashed_password
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    except:
        return False

def load_users():
    if Path(USER_DB).exists():
        with open(USER_DB, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DB, 'w') as f:
        json.dump(users, f, indent=2)

st.set_page_config(page_title="Test Auth", page_icon="🔐", layout="wide")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None

if not st.session_state.authenticated:
    st.title("TruthGuard AI")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            users = load_users()
            if username == ADMIN_USER and password == ADMIN_PASS:
                st.session_state.authenticated = True
                st.session_state.username = ADMIN_USER
                st.session_state.role = "admin"
                st.rerun()
            elif username in users and verify_password(password, users[username]["password"]):
                if len(users[username]["password"]) == 64:
                    users[username]["password"] = get_password_hash(password)
                    save_users(users)
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = users[username].get("role", "user")
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()
else:
    st.write(f"Logged in as {st.session_state.username} ({st.session_state.role})")