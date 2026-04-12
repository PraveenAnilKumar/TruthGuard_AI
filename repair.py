
import os

app_path = r'd:\TruthGuard_AI\app.py'

with open(app_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Reconstruct the file
# We want to keep everything up to the line where it breaks.
# The previous view showed line 1293 (authenticated tabs) was okay.
# Line 1294 was: login_tab, register_tab = st.tabs(["Login", "Register"])
# Line 1295 was the start of the broken section.

# I'll look for the st.tabs line.
start_index = -1
for i, line in enumerate(lines):
    if 'login_tab, register_tab = st.tabs(["Login", "Register"])' in line:
        start_index = i
        break

if start_index == -1:
    print("Could not find start point")
    exit(1)

# We want to keep everything up to start_index + 1
prefix = lines[:start_index + 1]

# Now we need the end point.
# The next reliable section is # SIDEBAR NAVIGATION
end_index = -1
for i, line in enumerate(lines):
    if '# SIDEBAR NAVIGATION' in line:
        end_index = i
        # Go back to the preceding header
        while end_index > 0 and '========' not in lines[end_index - 1]:
            end_index -= 1
        if end_index > 0:
            end_index -= 1 # Keep the header
        break

if end_index == -1:
    print("Could not find end point")
    exit(1)

suffix = lines[end_index:]

# The missing content
middle = [
    '\n',
    '    with login_tab:\n',
    '        with st.form("login_form"):\n',
    '            username = st.text_input("Username")\n',
    '            password = st.text_input("Password", type="password")\n',
    '            submitted = st.form_submit_button("Login", use_container_width=True)\n',
    '            if submitted:\n',
    '                users = load_users()\n',
    '                if username == ADMIN_USER and password == ADMIN_PASS and ADMIN_PASS:\n',
    '                    st.session_state.authenticated = True\n',
    '                    st.session_state.username = ADMIN_USER\n',
    '                    st.session_state.role = "admin"\n',
    '                    st.rerun()\n',
    '                elif username in users and verify_password(password, users[username]["password"]):\n',
    '                    if len(users[username]["password"]) == 64: \n',
    '                        users[username]["password"] = get_password_hash(password)\n',
    '                        save_users(users)\n',
    '                    st.session_state.authenticated = True\n',
    '                    st.session_state.username = username\n',
    '                    st.session_state.role = users[username].get("role", "user")\n',
    '                    st.rerun()\n',
    '                else:\n',
    '                    st.error("Invalid credentials.")\n',
    '\n',
    '    with register_tab:\n',
    '        with st.form("register_form"):\n',
    '            new_username = st.text_input("New username")\n',
    '            new_password = st.text_input("New password", type="password")\n',
    '            confirm_password = st.text_input("Confirm password", type="password")\n',
    '            create_admin = st.checkbox("Create as admin")\n',
    '            admin_key = st.text_input("Admin registration key", type="password", disabled=not create_admin)\n',
    '            registered = st.form_submit_button("Create account", use_container_width=True)\n',
    '\n',
    '            if registered:\n',
    '                if new_password != confirm_password:\n',
    '                    st.error("Passwords do not match.")\n',
    '                elif create_admin and admin_key != ADMIN_REG_KEY:\n',
    '                    st.error("Invalid admin registration key.")\n',
    '                else:\n',
    '                    role = "admin" if create_admin else "user"\n',
    '                    ok, message = register_user(new_username, new_password, role=role)\n',
    '                    if ok:\n',
    '                        st.success(message)\n',
    '                    else:\n',
    '                        st.error(message)\n',
    '    st.stop()\n',
    '\n',
    '# ============================================================================\n',
    '# LOAD DETECTORS (only after login - keeps startup fast)\n',
    '# ============================================================================\n',
    'deepfake_detector = None\n',
    'fake_news_detector = None\n',
    'sentiment_analyzer = None\n',
    'toxicity_detector = None\n',
    'realtime_verifier = None\n',
    'DEEPFAKE_AVAILABLE = False\n',
    'FAKE_NEWS_AVAILABLE = False\n',
    'SENTIMENT_AVAILABLE = False\n',
    'TOXICITY_AVAILABLE = False\n',
    'REALTIME_AVAILABLE = False\n',
    'ASPECT_AVAILABLE = False\n',
    'VIZ_AVAILABLE = False\n',
    'AspectSentimentAnalyzer = None\n',
    'SentimentVisualizer = None\n',
    'ToxicityVisualizer = None\n',
    '\n'
]

new_content = prefix + middle + suffix

with open(app_path, 'w', encoding='utf-8') as f:
    f.writelines(new_content)

print("Successfully repaired app.py")
