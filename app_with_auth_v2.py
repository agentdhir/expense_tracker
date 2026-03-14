import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import pickle
import firebase_admin
from firebase_admin import credentials, firestore
import time
import hashlib
import io
from threading import Thread
import numpy as np

# PAGE CONFIG
st.set_page_config(
    page_title="SpendSmart Tracker",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
    <style>
    .main { padding: 1rem 2rem; }
    h1, h2, h3 { color: #1f77b4; margin-bottom: 1.5rem; }
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border: none;
    }
    .stButton > button:hover {
        background-color: #155fa3;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# FIREBASE INITIALIZATION
# ==========================================
@st.cache_resource
def init_firebase():
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate('serviceAccountKey.json')
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"❌ Firebase Error: {str(e)}")
        st.stop()

db = init_firebase()

# ==========================================
# LOAD PRE-TRAINED MODELS
# ==========================================
@st.cache_resource
def load_baseline_models():
    """Pre-trained models load करो"""
    try:
        with open('models/decision_tree_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/le_cat.pkl', 'rb') as f:
            le_cat = pickle.load(f)
        with open('models/le_subcat.pkl', 'rb') as f:
            le_subcat = pickle.load(f)
        with open('models/le_advice.pkl', 'rb') as f:
            le_advice = pickle.load(f)
        
        return model, le_cat, le_subcat, le_advice
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None

@st.cache_resource
def load_user_model(username):
    """User-specific model load करो (agar retrained है)"""
    try:
        path = f'models/{username}'
        if os.path.exists(f'{path}/model.pkl'):
            with open(f'{path}/model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open(f'{path}/le_cat.pkl', 'rb') as f:
                le_cat = pickle.load(f)
            with open(f'{path}/le_subcat.pkl', 'rb') as f:
                le_subcat = pickle.load(f)
            with open(f'{path}/le_advice.pkl', 'rb') as f:
                le_advice = pickle.load(f)
            return model, le_cat, le_subcat, le_advice, True
    except:
        pass
    
    return None, None, None, None, False

# ==========================================
# AUTHENTICATION
# ==========================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, email, password):
    try:
        existing = db.collection('users').where('username', '==', username).stream()
        if list(existing):
            return False, "❌ Username already exists!"
        
        user_data = {
            'username': username,
            'email': email,
            'password': hash_password(password),
            'created_at': datetime.now(),
            'data_loaded': False,
            'model_retrained': False,
            'model_accuracy': 0.0
        }
        
        db.collection('users').document(username).set(user_data)
        return True, "✅ Registration successful!"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def login_user(username, password):
    try:
        user_doc = db.collection('users').document(username).get()
        
        if not user_doc.exists:
            return False, "❌ Username not found!"
        
        user_data = user_doc.to_dict()
        
        if user_data['password'] != hash_password(password):
            return False, "❌ Wrong password!"
        
        return True, "✅ Login successful!"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def load_csv_to_firebase(username):
    try:
        if not os.path.exists('student_expense_dataset.csv'):
            return False
        
        df = pd.read_csv('student_expense_dataset.csv')
        
        for index, row in df.iterrows():
            try:
                db.collection('expenses').add({
                    'date': str(row['date']),
                    'amount': float(row['amount']),
                    'category': str(row['category']),
                    'subcategory': str(row['subcategory']),
                    'description': '',
                    'spending_advice': str(row['spending_advice']),
                    'timestamp': datetime.now(),
                    'user_id': username
                })
            except:
                continue
        
        db.collection('users').document(username).update({'data_loaded': True})
        return True
    except:
        return False

def get_user_expenses(username):
    try:
        docs = db.collection('expenses').where('user_id', '==', username).stream()
        
        expenses = []
        for doc in docs:
            expenses.append(doc.to_dict())
        
        if expenses:
            expenses = sorted(expenses, key=lambda x: x.get('timestamp', datetime.now()), reverse=True)
            df = pd.DataFrame(expenses)
            df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y", errors='coerce')
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def retrain_model_background(username):
    """Background में user के लिए model retrain करो"""
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        docs = db.collection('expenses').where('user_id', '==', username).stream()
        expenses = [doc.to_dict() for doc in docs]
        
        if len(expenses) >= 20:
            df = pd.DataFrame(expenses)
            df_train = df.dropna(subset=['spending_advice']).copy()
            
            le_cat = LabelEncoder()
            le_subcat = LabelEncoder()
            le_advice = LabelEncoder()
            
            df_train["cat_encoded"] = le_cat.fit_transform(df_train["category"])
            df_train["subcat_encoded"] = le_subcat.fit_transform(df_train["subcategory"])
            df_train["advice_encoded"] = le_advice.fit_transform(df_train["spending_advice"])
            
            X = df_train[["amount", "cat_encoded", "subcat_encoded"]]
            y = df_train["advice_encoded"]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = DecisionTreeClassifier(random_state=42, max_depth=5)
            model.fit(X_train, y_train)
            
            test_accuracy = model.score(X_test, y_test)
            
            # Save
            os.makedirs(f'models/{username}', exist_ok=True)
            
            with open(f'models/{username}/model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open(f'models/{username}/le_cat.pkl', 'wb') as f:
                pickle.dump(le_cat, f)
            with open(f'models/{username}/le_subcat.pkl', 'wb') as f:
                pickle.dump(le_subcat, f)
            with open(f'models/{username}/le_advice.pkl', 'wb') as f:
                pickle.dump(le_advice, f)
            
            db.collection('users').document(username).update({
                'model_retrained': True,
                'model_accuracy': float(test_accuracy),
                'model_last_retrained': datetime.now()
            })
    except:
        pass

# ==========================================
# SESSION STATE
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.expense_data = None
    st.session_state.using_user_model = False

# ==========================================
# AUTH PAGE
# ==========================================
if not st.session_state.logged_in:
    st.title("🔐 SpendSmart Login/Register")
    
    auth_tab1, auth_tab2 = st.tabs(["🔓 Login", "📝 Register"])
    
    with auth_tab1:
        st.subheader("Login to Your Account")
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("🔓 Login", use_container_width=True):
            if username and password:
                success, message = login_user(username, password)
                
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    
                    user_expenses = get_user_expenses(username)
                    
                    if user_expenses.empty:
                        st.info("📚 Loading CSV data...")
                        if load_csv_to_firebase(username):
                            st.success("✅ Data loaded!")
                            time.sleep(1)
                            user_expenses = get_user_expenses(username)
                    
                    st.session_state.expense_data = user_expenses
                    st.success(message)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(message)
    
    with auth_tab2:
        st.subheader("Create New Account")
        
        reg_username = st.text_input("Username", key="reg_username")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm")
        
        if st.button("📝 Register", use_container_width=True):
            if not reg_username or not reg_email or not reg_password:
                st.warning("⚠️ Fill all fields!")
            elif reg_password != reg_password_confirm:
                st.error("❌ Passwords don't match!")
            elif len(reg_password) < 6:
                st.error("❌ Password too short!")
            else:
                success, message = register_user(reg_username, reg_email, reg_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    st.stop()

# ==========================================
# MAIN APP
# ==========================================

with st.sidebar:
    st.write(f"👤 **{st.session_state.username}**")
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.expense_data = None
        st.rerun()

st.title("💸 SpendSmart: Personal Expense Tracker")
st.markdown(f"*Welcome, **{st.session_state.username}**!*")

# Load models
model, le_cat, le_subcat, le_advice = load_baseline_models()
user_model, u_le_cat, u_le_subcat, u_le_advice, using_user_model = load_user_model(st.session_state.username)

if using_user_model:
    model = user_model
    le_cat = u_le_cat
    le_subcat = u_le_subcat
    le_advice = u_le_advice
    st.session_state.using_user_model = True

st.divider()

# Model status
col_status1, col_status2 = st.columns(2)
with col_status1:
    if st.session_state.using_user_model:
        st.success("✅ Using personalized model")
    else:
        st.info("📊 Using baseline model")

with col_status2:
    if model:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model failed")

st.divider()

# Load data
if st.session_state.expense_data is None:
    st.session_state.expense_data = get_user_expenses(st.session_state.username)

df = st.session_state.expense_data

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "➕ Add Expense",
    "📋 All Expenses",
    "🎯 Savings Goal",
    "📈 Full Report"
])

# ==========================================
# TAB 1: DASHBOARD
# ==========================================
with tab1:
    st.header("Your Financial Overview")
    
    if not df.empty:
        total_spent = df["amount"].sum()
        avg_transaction = df["amount"].mean()
        top_category = df.groupby("category")["amount"].sum().idxmax()
        transaction_count = len(df)
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("💰 Total Spent", f"₹{total_spent:,.0f}")
        with metric_col2:
            st.metric("📈 Avg. Expense", f"₹{avg_transaction:,.0f}")
        with metric_col3:
            st.metric("🏆 Top Category", top_category)
        with metric_col4:
            st.metric("📝 Transactions", transaction_count)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Spending by Category")
            cat_summary = df.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)
            fig_bar = px.bar(cat_summary, x="category", y="amount", color="amount", color_continuous_scale="Blues")
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.subheader("Daily Spending Trend")
            daily_summary = df.groupby("date")["amount"].sum().reset_index().sort_values('date')
            fig_line = px.line(daily_summary, x="date", y="amount", markers=True, line_shape="spline")
            fig_line.update_traces(line_color="#1f77b4", marker_size=6)
            fig_line.update_layout(height=400)
            st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("📭 No expenses yet!")

# ==========================================
# TAB 2: ADD EXPENSE
# ==========================================
with tab2:
    st.header("Add New Expense")
    
    if model is None:
        st.error("❌ Model not loaded!")
    else:
        # Load categories
        try:
            sample_df = pd.read_csv('student_expense_dataset.csv')
            cat_list = sorted(sample_df['category'].unique().tolist())
            subcat_dict = {}
            for cat in cat_list:
                subcat_dict[cat] = sorted(sample_df[sample_df['category'] == cat]['subcategory'].unique().tolist())
        except:
            cat_list = ["Food", "Travel", "Shopping", "Other"]
            subcat_dict = {cat: ["Other"] for cat in cat_list}
        
        with st.form("expense_form", clear_on_submit=True):
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                date_input = st.date_input("📅 Date", datetime.today())
                amount_input = st.number_input("💵 Amount (₹)", min_value=1, value=150, step=10)
            
            with form_col2:
                category_input = st.selectbox("📂 Category", cat_list, key="cat_select")
                subcat_list = subcat_dict.get(category_input, ["Other"])
                subcategory_input = st.selectbox("🏷️ Subcategory", subcat_list, key="subcat_select")
            
            description_input = st.text_area("📝 Description", placeholder="Details...")
            
            col_submit, col_space = st.columns([1, 4])
            with col_submit:
                submit_expense = st.form_submit_button("✨ Save & Predict", use_container_width=True)
        
        if submit_expense:
            try:
                with st.spinner("🤖 Making prediction..."):
                    cat_enc = le_cat.transform([category_input])[0]
                    subcat_enc = le_subcat.transform([subcategory_input])[0]
                    pred_enc = model.predict([[amount_input, cat_enc, subcat_enc]])
                    predicted_advice = le_advice.inverse_transform(pred_enc)[0]
                
                st.divider()
                
                if predicted_advice == "Reduce":
                    st.error("🛑 REDUCE - High/unnecessary expense")
                elif predicted_advice == "Acceptable":
                    st.warning("⚠️ ACCEPTABLE - Fits budget")
                else:
                    st.success("✅ ESSENTIAL - Important expense")
                
                st.info(f"**₹{amount_input}** | {category_input} > {subcategory_input}")
                
                st.divider()
                
                with st.spinner("☁️ Saving to Firebase..."):
                    db.collection('expenses').add({
                        'date': date_input.strftime("%d-%m-%Y"),
                        'amount': float(amount_input),
                        'category': category_input,
                        'subcategory': subcategory_input,
                        'description': description_input,
                        'spending_advice': predicted_advice,
                        'timestamp': datetime.now(),
                        'user_id': st.session_state.username
                    })
                    
                    # Retrain in background
                    Thread(target=retrain_model_background, args=(st.session_state.username,), daemon=True).start()
                    
                    st.success("✅ Saved!")
                    st.balloons()
                    
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    
                    time.sleep(2)
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ==========================================
# TAB 3: ALL EXPENSES
# ==========================================
with tab3:
    st.header("All Your Expenses")
    
    if not df.empty:
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            selected_category = st.multiselect(
                "Filter by Category",
                options=sorted(df['category'].unique()),
                default=sorted(df['category'].unique())
            )
        
        with col_filter2:
            selected_advice = st.multiselect(
                "Filter by AI Advice",
                options=sorted(df['spending_advice'].unique()),
                default=sorted(df['spending_advice'].unique())
            )
        
        filtered_df = df[
            (df['category'].isin(selected_category)) &
            (df['spending_advice'].isin(selected_advice))
        ].sort_values('date', ascending=False)
        
        st.subheader(f"📊 {len(filtered_df)} expenses")
        
        display_df = filtered_df[['date', 'amount', 'category', 'subcategory', 'spending_advice']].copy()
        display_df['date'] = display_df['date'].dt.strftime("%d-%m-%Y")
        display_df['amount'] = display_df['amount'].apply(lambda x: f"₹{x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, f"expenses_{st.session_state.username}.csv", "text/csv", use_container_width=True)
    else:
        st.info("📭 No expenses!")

# ==========================================
# TAB 4: SAVINGS GOAL
# ==========================================
with tab4:
    st.header("🎯 Savings Goal")
    
    monthly_income = st.number_input("Monthly Income (₹)", value=10000, step=500)
    goal_amount = st.number_input("Target Amount (₹)", value=45000, step=1000)
    months = st.slider("Months", 1, 24, 6)
    
    if st.button("Calculate", use_container_width=True):
        if not df.empty:
            monthly_spend = df["amount"].sum() / max(1, df["date"].dt.to_period("M").nunique())
            can_save = monthly_income - monthly_spend
            need_save = goal_amount / months
            gap = need_save - can_save
            
            st.metric("Monthly Spend", f"₹{monthly_spend:,.0f}")
            st.metric("Can Save", f"₹{can_save:,.0f}")
            
            if gap > 0:
                st.warning(f"Gap: ₹{gap:,.0f}/month")
            else:
                st.success(f"✅ Achievable in {months} months!")

# ==========================================
# TAB 5: FULL REPORT
# ==========================================
with tab5:
    st.header("📈 Full Report")
    
    if not df.empty:
        if st.button("📊 Generate Report", use_container_width=True):
            st.markdown(f"# Report - {st.session_state.username}")
            st.markdown(f"**Generated:** {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
            st.divider()
            
            st.metric("Total Spent", f"₹{df['amount'].sum():,.0f}")
            st.metric("Avg Expense", f"₹{df['amount'].mean():,.0f}")
            st.metric("Transactions", len(df))
            
            st.divider()
            
            cat_summary = df.groupby('category')['amount'].sum()
            fig = px.pie(values=cat_summary.values, names=cat_summary.index, hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download", csv, f"report_{st.session_state.username}.csv", "text/csv", use_container_width=True)

st.divider()
st.markdown("**SpendSmart** - Made with ❤️ using Streamlit & Firebase | Version 1.0")