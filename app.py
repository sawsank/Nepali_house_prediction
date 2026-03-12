import streamlit as st
from src.agent import get_agent
from src.ml_model import get_ml_predictor
from src.eda_logic import get_eda_plots
from src.vector_db import add_document_to_db
import pandas as pd

# Page configuration
st.set_page_config(page_title="Agentic RAG AI - Real Estate Nepal", page_icon="🏠", layout="wide")

# Custom CSS (Premium Light Theme)
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    .main-header {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 30px;
        color: white !important;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .main-header h1, .main-header p {
        color: white !important;
    }
    .stSidebar {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    div[data-testid="stMetricValue"] {
        color: #2563eb;
    }
    /* Sidebar Navigation Highlighted Buttons */
    [data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    /* Definitive fix: Hide the radio circle icon and its wrapper */
    [data-testid="stRadio"] div[role="radiogroup"] [data-testid="stRadioDot"] {
        display: none !important;
    }
    [data-testid="stRadio"] div[role="radiogroup"] > div > div:first-child {
        display: none !important;
    }
    /* Style the option container */
    [data-testid="stRadio"] div[role="radiogroup"] > div {
        background-color: transparent !important;
        padding: 0 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    /* Active Highlight */
    [data-testid="stRadio"] div[role="radiogroup"] > div[data-checked="true"] {
        background-color: #2563eb !important;
    }
    /* Hover effect */
    [data-testid="stRadio"] div[role="radiogroup"] > div:hover {
        background-color: #f1f5f9;
    }
    [data-testid="stRadio"] div[role="radiogroup"] > div[data-checked="true"]:hover {
        background-color: #1e40af !important;
    }
    /* Label/Text Styling */
    [data-testid="stRadio"] label {
        padding: 12px 16px !important;
        margin: 0 !important;
        width: 100% !important;
        cursor: pointer !important;
        color: #475569 !important;
    }
    [data-testid="stRadio"] div[data-checked="true"] label p {
        color: white !important;
        font-weight: 600 !important;
    }
    /* Suggestion Chips */
    .suggestion-chip {
        display: inline-block;
        background-color: #e2e8f0;
        padding: 5px 12px;
        border-radius: 15px;
        margin: 5px;
        font-size: 0.85rem;
        cursor: pointer;
        border: 1px solid #cbd5e1;
    }
    .suggestion-chip:hover {
        background-color: #3b82f6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def display_agent_reasoning(steps):
    """Helper to display agent reasoning with source indicators."""
    with st.expander("🔍 View Agent Reasoning", expanded=False):
        for action, observation in steps:
            # Identify the tool source
            tool_name = getattr(action, 'tool', 'None')
            if tool_name == "HouseKnowledgeBase":
                source_label = "🏠 **Source:** Local Knowledge Base"
            elif tool_name == "duckduckgo_search":
                source_label = "🌐 **Source:** Web Search (Real-time)"
            else:
                source_label = f"🛠️ **Tool:** {tool_name}"
            
            st.info(source_label)
            st.markdown(f"**Thought:** {getattr(action, 'log', 'Thinking...')}")
            with st.expander("📄 View Tool Output", expanded=False):
                st.markdown(observation)

# Navigation
st.sidebar.title("🏢 Navigation")
page = st.sidebar.radio(
    "Go to:", 
    ["💬 Agentic Chat", "📊 Price Predictor (ML)", "📈 Data Insights (EDA)", "📥 Add Knowledge"],
    label_visibility="collapsed"
)

st.sidebar.divider()
if st.sidebar.button("🗑️ Clear Chat History", width="stretch"):
    st.session_state.messages = []
    if "agent" in st.session_state:
        with st.spinner("Resetting Agent..."):
            st.session_state.agent = get_agent()
    st.success("Chat history cleared!")
    st.rerun()

if page == "💬 Agentic Chat":
    st.markdown('<div class="main-header"><h1>🏠 Agentic Chat Explorer</h1><p>Query the house database or search the web.</p></div>', unsafe_allow_html=True)

    if "agent" not in st.session_state:
        with st.spinner("Initializing Agentic System..."):
            st.session_state.agent = get_agent()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Suggested Questions Section
    st.info("💡 **Suggested Questions: (Click to ask)**")
    suggestions = [
        "What is the average price of houses in Imadol?",
        "Find me a 4-bedroom house in Lalitpur.",
        "What are common amenities in Nepal real estate?",
        "Search online for 2024 house tax rates in Kathmandu.",
        "Compare prices between Bhaisepati and Budhanilkantha."
    ]
    
    # Store the clicked suggestion in session state to trigger the search
    clicked_suggestion = None
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, key=f"sug_{i}", width="stretch"):
            clicked_suggestion = suggestion

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "reasoning" in message and message["reasoning"]:
                display_agent_reasoning(message["reasoning"])

    # Handle Input (either from chat_input or a suggestion click)
    prompt = st.chat_input("Ask me about real estate in Nepal...")
    
    # If a suggestion was clicked, override the prompt
    final_prompt = clicked_suggestion if clicked_suggestion else prompt

    if final_prompt:
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with st.chat_message("user"):
            st.markdown(final_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response_data = st.session_state.agent.invoke({"input": final_prompt})
                    
                    # Extract Reasoning / Intermediate Steps
                    if "intermediate_steps" in response_data:
                        display_agent_reasoning(response_data["intermediate_steps"])
                    
                    response = response_data["output"]
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "reasoning": response_data.get("intermediate_steps", [])
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    # Debugging info
                    st.exception(e)

elif page == "📊 Price Predictor (ML)":
    st.markdown('<div class="main-header"><h1>📊 Machine Learning Prediction</h1><p>Predict house prices based on comprehensive features.</p></div>', unsafe_allow_html=True)

    if "predictor" not in st.session_state:
        with st.spinner("Training Model..."):
            predictor, success, msg = get_ml_predictor()
            if success:
                st.session_state.predictor = predictor
                st.success(msg)
            else:
                st.error(f"Training failed: {msg}")

    if "predictor" in st.session_state:
        predictor = st.session_state.predictor
        
        # Initialize session state for prediction result
        if "prediction_result" not in st.session_state:
            st.session_state.prediction_result = None

        # --- Case 1: Results Page (Hide Form) ---
        if st.session_state.prediction_result:
            res = st.session_state.prediction_result
            st.success("✅ Prediction Generated!")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Estimated Price", f"Rs. {res['cr_val']:.2f} Cr")
                st.info(f"Approx: Rs. {res['prediction']:,.0f}")
            
            with col2:
                if st.button("🔄 Predict Another Property", width="stretch"):
                    st.session_state.prediction_result = None
                    st.rerun()

            # --- AI Market Reasoning (Fetch if not already present) ---
            if "ai_output" not in res:
                with st.spinner("AI is analyzing market reasoning..."):
                    try:
                        features_summary = f"Location: {res['inputs']['Location']}, Facing: {res['inputs']['Facing']}, Land Area: {res['inputs']['Land Area']} aana, Floors: {res['inputs']['Floors']}, Bedrooms: {res['inputs']['Bedrooms']}, Bathrooms: {res['inputs']['Bathrooms']}, Road Access: {res['inputs']['Road Access']} ft."
                        reasoning_prompt = f"The ML model predicts a house price of Rs. {res['cr_val']:.2f} Crore for a property with these features: {features_summary}. Provide a short market reasoning and context for why this price makes sense in the current Nepal real estate market."
                        
                        if "agent" not in st.session_state:
                            st.session_state.agent = get_agent()
                            
                        reasoning_resp = st.session_state.agent.invoke({"input": reasoning_prompt})
                        
                        st.session_state.prediction_result["ai_output"] = reasoning_resp["output"]
                        st.session_state.prediction_result["reasoning"] = reasoning_resp.get("intermediate_steps", [])
                        st.rerun() # Refresh to show results
                    except Exception as e:
                        st.session_state.prediction_result["ai_output"] = "AI Insights currently unavailable."
                        st.session_state.prediction_result["reasoning"] = []
                        st.rerun()
            
            # Display AI Output once loaded
            if "ai_output" in res:
                st.subheader("💡 AI Market Insights")
                
                st.markdown(res['ai_output'])
                
                if res.get('reasoning'):
                    display_agent_reasoning(res['reasoning'])
            
            st.divider()
            st.subheader("📋 Property Details Summary")
            
            # Professional Grid for Details
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown(f"📍 **Location:** {res['inputs']['Location']}")
                st.markdown(f"🧭 **Facing:** {res['inputs']['Facing']}")
            with d2:
                st.markdown(f"📐 **Land Area:** {res['inputs']['Land Area']} aana")
                st.markdown(f"🛣️ **Road Access:** {res['inputs']['Road Access']} ft.")
            with d3:
                st.markdown(f"🏢 **Floors:** {res['inputs']['Floors']}")
                st.markdown(f"🛏️ **Bed/Bath:** {res['inputs']['Bedrooms']} / {res['inputs']['Bathrooms']}")

        # --- Case 2: Input Form ---
        else:
            with st.form("prediction_form"):
                st.subheader("🏠 Property Details")
                col1, col2 = st.columns(2)
                with col1:
                    location = st.selectbox("Select Location", options=predictor.locations)
                    facing = st.selectbox("Facing Direction", options=predictor.facings)
                    land_area = st.number_input("Land Area (aana)", min_value=1.0, value=4.0)
                    road_access = st.slider("Road Access (Feet)", 0, 50, 12)
                with col2:
                    floors = st.number_input("Floors", 1.0, 10.0, 2.5)
                    bedrooms = st.number_input("Bedrooms", 1, 20, 4)
                    bathrooms = st.number_input("Bathrooms", 1, 20, 3)
                    amenities_list = ["Drinking Water", "Parking", "Garden", "Drainage", "Solar"]
                    selected_amenities = st.multiselect("Amenities", options=amenities_list)
                
                submit = st.form_submit_button("🚀 Predict Price", width="stretch")

            if submit:
                prediction = predictor.predict(land_area, road_access, floors, bedrooms, bathrooms, location, facing, len(selected_amenities))
                if prediction:
                    cr_val = prediction / 10000000
                    # Store basic results and rerun immediately to hide the form
                    st.session_state.prediction_result = {
                        "prediction": prediction,
                        "cr_val": cr_val,
                        "inputs": {
                            "Location": location,
                            "Facing": facing,
                            "Land Area": land_area,
                            "Floors": floors,
                            "Bedrooms": bedrooms,
                            "Bathrooms": bathrooms,
                            "Road Access": road_access
                        },
                        "best_model": predictor.best_model_name
                    }
                    st.rerun()
                else:
                    st.error("Prediction unavailable for these inputs.")

elif page == "📈 Data Insights (EDA)":
    st.markdown('<div class="main-header"><h1>📈 Data Insights (EDA)</h1><p>Visualizing market trends and property statistics.</p></div>', unsafe_allow_html=True)
    
    # Pass ML metrics if available
    ml_metrics = st.session_state.predictor.metrics if "predictor" in st.session_state else None
    
    with st.spinner("Analyzing data..."):
        plots = get_eda_plots(ml_metrics=ml_metrics)
    
    if plots:
        # --- Metrics Row ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Listings", plots["stats"]["total"])
        m2.metric("Avg Price", f"Rs. {plots['stats']['avg']:.2f} Cr")
        m3.metric("Max Price", f"Rs. {plots['stats']['max']:.2f} Cr")
        
        st.divider()

        # --- Visualizations with Descriptions ---
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plots["dist"], use_container_width=True)
            st.markdown("**Price Distribution:** This histogram shows how house prices are distributed across the dataset. Most listings fall within a specific range, highlighting the most active market segment.")
        
        with c2:
            st.plotly_chart(plots["loc"], use_container_width=True)
            st.markdown("**Top Locations by Volume:** This chart identifies which areas in Nepal have the highest number of property listings, indicating where the market is most liquid and competitive.")
        
        st.divider()
        
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plots["avg_price"], use_container_width=True)
            st.markdown("**Most Expensive Locations:** Shows the top areas with the highest average house prices. This reflects premium demand and high-value residential hubs.")
        
        with c4:
            st.plotly_chart(plots["corr"], use_container_width=True)
            st.markdown("**Feature Correlation Heatmap:** This heatmap visualizes how strongly different property features (like floors, bedrooms, and bathrooms) relate to the total price.")

        if plots.get("model_comp"):
            st.divider()
            st.plotly_chart(plots["model_comp"], use_container_width=True)
            
            best_model_name = st.session_state.predictor.best_model_name
            st.markdown(f"""
            ### 🏅 The "AI Battle" Results
            To provide the most reliable price estimates, the system triggers an **automated tournament** between four distinct machine learning algorithms. 
            
            **The selected Champion is: {best_model_name}**
            
            **Why this model won:**
            - **R² Score (Accuracy)**: This model achieved the highest Coefficient of Determination ({st.session_state.predictor.metrics[best_model_name]:.4f}). In data science, this means it best captures the complex relationship between land area, location, and price.
            - **Error Minimization**: It demonstrated the lowest average error on the historical Nepal property dataset, making it the most trustworthy "brain" for your search.
            """)

        st.divider()
        
        # --- AI Market Analysis Section (Moved to Bottom) ---
        st.subheader("💡 AI Market Analysis")
        with st.expander("Click to view AI interpretation of the visual data shown above", expanded=True):
            with st.spinner("AI is interpreting market trends..."):
                try:
                    if "agent" not in st.session_state:
                        st.session_state.agent = get_agent()
                    
                    stats = plots["stats"]
                    eda_summary = f"""
                    Total Property Listings: {stats['total']}
                    Average House Price: Rs. {stats['avg']:.2f} Cr
                    Maximum Price Found: Rs. {stats['max']:.2f} Cr
                    Minimum Price Found: Rs. {stats['min']:.2f} Cr
                    """
                    eda_prompt = f"Analyze these real estate market statistics and the visual trends shown in the charts above. Provide 3-4 professional insights about the current trends, demand, and volume in Nepal: {eda_summary}"
                    
                    eda_ai_resp = st.session_state.agent.invoke({"input": eda_prompt})
                    st.markdown(eda_ai_resp["output"])
                    
                    if "intermediate_steps" in eda_ai_resp:
                        display_agent_reasoning(eda_ai_resp["intermediate_steps"])
                except Exception as e:
                    st.warning("AI Analysis currently unavailable.")

elif page == "📥 Add Knowledge":
    st.markdown('<div class="main-header"><h1>📥 Add Knowledge</h1><p>Expand the AI knowledge base manually.</p></div>', unsafe_allow_html=True)
    
    # Success message persistence across rerun
    if "kb_success_msg" in st.session_state:
        st.success(st.session_state.kb_success_msg)
        del st.session_state.kb_success_msg

    with st.form("knowledge_form", clear_on_submit=True):
        category = st.selectbox("Knowledge Category", ["Market Trends", "Rules & Regulations", "Specific Property", "Agent Tip", "Other"])
        content = st.text_area("What would you like the AI to remember?", placeholder="Example: In 2024, the property tax in Kathmandu ward 10 increased by 5%.", height=150)
        
        if st.form_submit_button("✅ Add to Knowledge Base", width="stretch"):
            if not content.strip():
                st.error("Please enter some meaningful content.")
            else:
                with st.spinner("Indexing new knowledge..."):
                    success, msg = add_document_to_db(content, category)
                    if success:
                        st.session_state.kb_success_msg = f"Successfully added: {msg}"
                        st.rerun()
                    else:
                        st.error(msg)
