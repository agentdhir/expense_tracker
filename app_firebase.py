import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import firebase_admin
from firebase_admin import credentials, firestore
import time

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
    .main { 
        padding: 1rem 2rem; 
    }
    
    h1, h2, h3 { 
        color: #1f77b4; 
        margin-bottom: 1.5rem; 
    }
    
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
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #155fa3;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }
    
    .prediction-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .reduce-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    
    .acceptable-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    
    .essential-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# FIREBASE INITIALIZATION
# ==========================================
@st.cache_resource
def init_firebase():
    """Firebase ko ek baar initialize karo"""
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate('serviceAccountKey.json')
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"❌ Firebase Error: {str(e)}")
        st.info("💡 Hint: serviceAccountKey.json file project folder mein hona chahiye!")
        st.stop()

db = init_firebase()

# ==========================================
# FIRESTORE SE DATA LOAD KARO
# ==========================================
@st.cache_data(ttl=60)
def load_expenses_from_firestore():
    """Firestore se expenses load karo"""
    try:
        docs = db.collection('expenses').order_by(
            'timestamp', 
            direction=firestore.Query.DESCENDING
        ).limit(100).stream()
        
        expenses = []
        for doc in docs:
            expenses.append(doc.to_dict())
        
        if expenses:
            df = pd.DataFrame(expenses)
            df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y", errors='coerce')
            return df
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠️ Firestore load error: {e}")
        return pd.DataFrame()

# Fallback: Local CSV se load karo
def load_data_local():
    """Agar Firestore fail ho to CSV se load karo"""
    DATA_FILE = "student_expense_dataset.csv"
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors='coerce')
        return df
    return pd.DataFrame()

# Data load karo
try:
    expenses_df = load_expenses_from_firestore()
    if expenses_df.empty:
        st.info("📭 Firestore mein abhi koi data nahi, local CSV load kar raha hoon...")
        expenses_df = load_data_local()
except Exception as e:
    st.warning(f"⚠️ Firebase se connect nahi hua: {e}")
    expenses_df = load_data_local()

# Session state mein store karo
if 'expense_data' not in st.session_state:
    st.session_state.expense_data = expenses_df

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

if 'saved_status' not in st.session_state:
    st.session_state.saved_status = None

# ==========================================
# FIRESTORE MEIN DATA SAVE KARO
# ==========================================
def save_expense_to_firestore(date, amount, category, subcategory, advice):
    """Expense ko Firestore mein save karo"""
    try:
        db.collection('expenses').add({
            'date': date.strftime("%d-%m-%Y"),
            'amount': float(amount),
            'category': category,
            'subcategory': subcategory,
            'spending_advice': advice,
            'timestamp': datetime.now(),
            'user_id': 'student'
        })
        
        # Cache clear karo taaki new data dikhe
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"❌ Save Error: {e}")
        return False

# ==========================================
# AI MODEL TRAINING
# ==========================================
@st.cache_resource
def train_ai_model(data):
    """Decision Tree model train karo"""
    if data.empty or 'spending_advice' not in data.columns:
        st.error("❌ Data nahi hai model train karne ke liye!")
        st.stop()
    
    df_train = data.dropna(subset=['spending_advice']).copy()
    
    if df_train.empty:
        st.error("❌ Valid training data nahi mil raha!")
        st.stop()
    
    le_cat = LabelEncoder()
    le_subcat = LabelEncoder()
    le_advice = LabelEncoder()
    
    df_train["cat_encoded"] = le_cat.fit_transform(df_train["category"])
    df_train["subcat_encoded"] = le_subcat.fit_transform(df_train["subcategory"])
    df_train["advice_encoded"] = le_advice.fit_transform(df_train["spending_advice"])
    
    X = df_train[["amount", "cat_encoded", "subcat_encoded"]]
    y = df_train["advice_encoded"]
    
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X, y)
    
    return model, le_cat, le_subcat, le_advice

# Model initialize karo
if not st.session_state.expense_data.empty:
    try:
        model, le_cat, le_subcat, le_advice = train_ai_model(st.session_state.expense_data)
    except Exception as e:
        st.error(f"❌ Model training error: {e}")
        st.stop()
else:
    st.error("❌ Koi data nahi hai! Pehle data add karo ya CSV load karo.")
    st.stop()

# ==========================================
# APP HEADER
# ==========================================
st.title("💸 SpendSmart: Personal Expense Tracker")
st.markdown("*Track your spending, get AI insights, and achieve your savings goals*")
st.markdown("🔥 **Connected to Firebase** | 📱 Mobile Ready | 🤖 AI Powered")
st.divider()

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "➕ Add Expense",
    "🎯 Savings Goal",
    "📂 Upload Report"
])

# ==========================================
# TAB 1: DASHBOARD
# ==========================================
with tab1:
    st.header("Your Financial Overview")
    df = st.session_state.expense_data
    
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
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Spending by Category")
            cat_summary = df.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)
            fig_bar = px.bar(
                cat_summary, 
                x="category", 
                y="amount",
                color="amount",
                color_continuous_scale="Blues",
                labels={"amount": "Amount (₹)", "category": "Category"}
            )
            fig_bar.update_layout(height=400, showlegend=False, hovermode='x unified')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.subheader("Daily Spending Trend")
            daily_summary = df.groupby("date")["amount"].sum().reset_index()
            fig_line = px.line(
                daily_summary,
                x="date",
                y="amount",
                markers=True,
                line_shape="spline",
                labels={"date": "Date", "amount": "Amount (₹)"}
            )
            fig_line.update_traces(line_color="#1f77b4", marker_size=6)
            fig_line.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_line, use_container_width=True)
        
        st.divider()
        
        # More insights
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            st.subheader("Category Breakdown")
            category_totals = df.groupby("category")["amount"].sum().sort_values(ascending=True)
            fig_pie = px.pie(
                values=category_totals.values,
                names=category_totals.index,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_insight2:
            st.subheader("Top Subcategories")
            top_subcat = df.groupby("subcategory")["amount"].sum().sort_values(ascending=False).head(8)
            fig_subcat = px.bar(
                x=top_subcat.values,
                y=top_subcat.index,
                orientation='h',
                color=top_subcat.values,
                color_continuous_scale="Viridis",
                labels={"x": "Amount (₹)"}
            )
            fig_subcat.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_subcat, use_container_width=True)
    else:
        st.info("📭 Koi expenses nahi. 'Add Expense' tab mein jao aur add karo!")

# ==========================================
# TAB 2: ADD EXPENSE
# ==========================================
with tab2:
    st.header("Add New Expense")
    st.markdown("Apna expense log karo aur AI advice paao (Firebase mein save hoga)")
    
    # Form
    with st.form("expense_form", clear_on_submit=False):
        form_col1, form_col2 = st.columns(2)
        
        with form_col1:
            date_input = st.date_input("📅 Date", datetime.today())
            amount_input = st.number_input("💵 Amount (₹)", min_value=1, value=150, step=10)
        
        with form_col2:
            cat_list = sorted(st.session_state.expense_data['category'].unique().tolist())
            category_input = st.selectbox("📂 Category", cat_list, key="cat_select")
            
            subcat_list = sorted(st.session_state.expense_data[
                st.session_state.expense_data['category'] == category_input
            ]['subcategory'].unique().tolist())
            subcategory_input = st.selectbox("🏷️ Subcategory", subcat_list, key="subcat_select")
        
        col_submit, col_space = st.columns([1, 4])
        with col_submit:
            submit_expense = st.form_submit_button("✨ Analyze & Save", use_container_width=True)
    
    # Results ko form se bahar rakho taaki persist rahe
    if submit_expense:
        try:
            # Loading indicator
            with st.spinner("🔄 AI analysis chal rahi hai..."):
                time.sleep(1)
                
                # Prediction karo
                cat_enc = le_cat.transform([category_input])[0]
                subcat_enc = le_subcat.transform([subcategory_input])[0]
                pred_enc = model.predict([[amount_input, cat_enc, subcat_enc]])
                predicted_advice = le_advice.inverse_transform(pred_enc)[0]
            
            st.divider()
            
            # Advice message
            if predicted_advice == "Reduce":
                st.markdown("""
                <div class='prediction-box reduce-box'>
                    <h3>🛑 REDUCE</h3>
                    <p><strong>High/unnecessary expense</strong></p>
                    <p>💡 Tip: Ye expense zaroori nahi hai, avoid karo!</p>
                </div>
                """, unsafe_allow_html=True)
                
            elif predicted_advice == "Acceptable":
                st.markdown("""
                <div class='prediction-box acceptable-box'>
                    <h3>⚠️ ACCEPTABLE</h3>
                    <p><strong>Fits budget, but be careful</strong></p>
                    <p>💡 Tip: Ye expense thik hai, lekin regularly mat karo!</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown("""
                <div class='prediction-box essential-box'>
                    <h3>✅ ESSENTIAL</h3>
                    <p><strong>Important expense</strong></p>
                    <p>💡 Tip: Ye expense zaroori hai, safely spend kar sakte ho!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Summary
            st.info(f"**Amount:** ₹{amount_input} | **Category:** {category_input} | **Subcategory:** {subcategory_input}")
            
            st.divider()
            
            # Firebase mein save karo
            with st.spinner("☁️ Firebase mein save ho raha hai..."):
                time.sleep(1)
                
                if save_expense_to_firestore(date_input, amount_input, category_input, 
                                            subcategory_input, predicted_advice):
                    st.success("✅ Firebase mein successfully save ho gaya!")
                    st.balloons()
                    
                    # Success details
                    col_success1, col_success2 = st.columns(2)
                    with col_success1:
                        st.metric("Status", "Saved ✓")
                    with col_success2:
                        st.metric("Prediction", predicted_advice)
                    
                else:
                    st.error("❌ Firebase mein save nahi hua!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info(f"💡 Debug info: {type(e).__name__}")

# ==========================================
# TAB 3: SAVINGS GOAL
# ==========================================
with tab3:
    st.header("🎯 Savings Goal Engine")
    st.markdown("Apna savings goal plan karo")
    
    col_goal1, col_goal2 = st.columns([1, 1])
    
    with col_goal1:
        st.subheader("Goal Parameters")
        monthly_income = st.number_input("Monthly Income/Allowance (₹)", value=10000, step=500)
        goal_name = st.text_input("Kya save kar rahe ho?", value="PlayStation 5")
        goal_amount = st.number_input("Target Amount (₹)", value=45000, step=1000)
        months_to_save = st.slider("Kitne mahine mein?", min_value=1, max_value=24, value=6)
        calculate_btn = st.button("Calculate Plan", use_container_width=True)
    
    with col_goal2:
        if calculate_btn:
            df = st.session_state.expense_data
            
            if df.empty:
                st.warning("⚠️ Koi data nahi! Pehle expenses add karo.")
            else:
                months_in_data = max(1, df["date"].dt.to_period("M").nunique())
                actual_monthly_spend = df["amount"].sum() / months_in_data
                current_saving_capacity = monthly_income - actual_monthly_spend
                required_monthly_saving = goal_amount / months_to_save
                saving_gap = required_monthly_saving - current_saving_capacity
                
                st.subheader(f"📊 Plan: {goal_name}")
                st.divider()
                
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Monthly Spend", f"₹{actual_monthly_spend:,.0f}")
                with stat_col2:
                    st.metric("Can Save", f"₹{current_saving_capacity:,.0f}")
                with stat_col3:
                    st.metric("Need/Month", f"₹{required_monthly_saving:,.0f}")
                
                st.divider()
                
                if saving_gap > 0:
                    st.warning(f"⚠️ **Gap: ₹{saving_gap:,.0f}/month**")
                    st.markdown(f"""
                    You need to save **₹{required_monthly_saving:,.0f}** per month, but can only save **₹{current_saving_capacity:,.0f}**.
                    
                    **Suggestions:**
                    - Reduce spending by ₹{saving_gap:,.0f}/month
                    - Extend timeline to {int(goal_amount / max(current_saving_capacity, 1))} months
                    - Increase income by ₹{saving_gap:,.0f}/month
                    """)
                else:
                    st.success(f"✅ **Goal achievable in {months_to_save} months!**")
                    st.markdown(f"""
                    You'll have **₹{goal_amount:,}** saved for **{goal_name}** in {months_to_save} months!
                    
                    Current plan: Save **₹{required_monthly_saving:,.0f}** per month
                    """)

# ==========================================
# TAB 4: UPLOAD REPORT
# ==========================================
with tab4:
    st.header("📂 Upload & Generate Report")
    st.markdown("Apni CSV file upload karo aur AI analysis paao")
    
    with st.expander("📋 Required CSV Format (Click to expand)", expanded=False):
        st.markdown("""
        Your CSV file **must** have these 4 columns (lowercase):
        - `date` (format: DD-MM-YYYY)
        - `amount` (numbers only)
        - `category` (text)
        - `subcategory` (text)
        """)
        
        sample_df = pd.DataFrame({
            "date": ["15-03-2026", "16-03-2026", "17-03-2026"],
            "amount": [150, 40, 500],
            "category": ["Food", "Travel", "Shopping"],
            "subcategory": ["Restaurant", "Bus/Metro", "Clothes"]
        })
        
        st.dataframe(sample_df, hide_index=True, use_container_width=True)
        
        template_csv = sample_df.iloc[0:0].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Template",
            data=template_csv,
            file_name="SpendSmart_Template.csv",
            mime="text/csv"
        )
    
    st.divider()
    uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        
        required_cols = ["date", "amount", "category", "subcategory"]
        if not all(col in user_df.columns for col in required_cols):
            st.error(f"❌ Missing columns! Required: {', '.join(required_cols)}")
        else:
            with st.spinner("🔄 AI is analyzing your data..."):
                predictions = []
                for index, row in user_df.iterrows():
                    try:
                        cat_enc = le_cat.transform([row['category']])[0]
                        subcat_enc = le_subcat.transform([row['subcategory']])[0]
                        pred_enc = model.predict([[row['amount'], cat_enc, subcat_enc]])
                        pred_advice = le_advice.inverse_transform(pred_enc)[0]
                        predictions.append(pred_advice)
                    except:
                        predictions.append("Acceptable (Unknown)")
                
                user_df["AI_spending_advice"] = predictions
                
                st.success("✅ Analysis Complete!")
                st.divider()
                
                # Summary stats
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    st.metric("Total Expenses", len(user_df))
                with summary_col2:
                    st.metric("Total Amount", f"₹{user_df['amount'].sum():,.0f}")
                with summary_col3:
                    reduce_count = (user_df['AI_spending_advice'] == 'Reduce').sum()
                    st.metric("⚠️ High Spends", reduce_count)
                
                st.divider()
                st.subheader("Detailed Report")
                st.dataframe(user_df, use_container_width=True, hide_index=True)
                
                csv_report = user_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Report",
                    data=csv_report,
                    file_name="SpendSmart_AI_Report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ==========================================
# Footer
# ==========================================
st.divider()
st.markdown("""
---
**SpendSmart** - Your Personal AI Finance Advisor
- 🔥 Firebase Connected | 📱 Mobile Ready | 🤖 AI Powered
- Made with ❤️ using Streamlit & Firebase
""")