"""
Streamlit web interface - FIXED VERSION
All prompts use safe, non-triggering language
"""
import streamlit as st
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.agent import CodeExplainerAgent, analyze_code_complexity
from core.config import AgentConfig, Language

def initialize_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        
        from dotenv import load_dotenv
        load_dotenv()
        
        st.session_state.provider = "gemini"
        st.session_state.model_name = "gemini-2.5-flash"
        st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")
        st.session_state.temperature = 0.2
        st.session_state.agent = None
        st.session_state.config = None

def get_agent():
    """Get or create the agent instance."""
    if st.session_state.agent is None:
        try:
            st.session_state.config = AgentConfig(
                project_root=Path(__file__).parent.absolute(),
                llm_provider=st.session_state.provider,
                llm_model=st.session_state.model_name,
                llm_api_key=st.session_state.api_key,
                temperature=st.session_state.temperature
            )
            st.session_state.agent = CodeExplainerAgent(st.session_state.config)
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            return None
    
    return st.session_state.agent

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Code Explainer Agent",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Code Explainer Agent")
    st.markdown("Analyze, explain, and improve your Python code with AI assistance")
    
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.session_state.api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.api_key,
            type="password",
            help="Your Google Gemini API key"
        )
        st.session_state.temperature = st.slider(
            "Temperature",
            0.0, 1.0, st.session_state.temperature,
            help="Lower = more deterministic, Higher = more creative"
        )
        
        if st.button("🔄 Reinitialize Agent"):
            st.session_state.agent = None
            st.session_state.config = None
            st.success("Agent reinitialized")
    
    # Main interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Analyze", "📚 Document", "✨ Improve", 
        "🔍 Explain", "⚖️ Compare"
    ])
    
    agent = get_agent()
    if agent is None:
        st.error("❌ Failed to initialize agent. Please check your API key.")
        return
    
    # Tab 1: Analyze Code
    with tab1:
        st.header("📝 Code Analysis")
        code_input = st.text_area(
            "Paste your Python code:",
            height=300,
            placeholder="def hello():\n    print('Hello, world!')"
        )
        
        if st.button("Analyze Code", key="analyze"):
            if code_input.strip():
                with st.spinner("Analyzing code..."):
                    result = agent.analyze_code(code_input)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📊 Complexity Metrics")
                        st.info(f"""
                        - **Lines**: {result['lines']}
                        - **Functions**: {result['functions']}
                        - **Classes**: {result['classes']}
                        - **Complexity**: {result['complexity'].upper()}
                        """)
                    
                    with col2:
                        st.subheader("🤖 AI Analysis")
                        st.markdown(result.get('ai_analysis', 'No analysis available'))
            else:
                st.warning("Please enter some code to analyze")
    
    # Tab 2: Generate Documentation
    with tab2:
        st.header("📚 Documentation Generator")
        code_for_docs = st.text_area(
            "Code to document:",
            height=300,
            key="doc_input",
            placeholder="def calculate(x, y):\n    return x + y"
        )
        
        if st.button("Generate Documentation", key="doc"):
            if code_for_docs.strip():
                with st.spinner("Generating documentation..."):
                    documentation = agent.generate_documentation(code_for_docs)
                    st.markdown(documentation)
                    
                    st.download_button(
                        "📥 Download Documentation",
                        data=documentation,
                        file_name="documentation.md",
                        mime="text/markdown"
                    )
            else:
                st.warning("Please enter code to document")
    
    # Tab 3: Suggest Improvements
    with tab3:
        st.header("✨ Improvement Suggestions")
        code_for_improvement = st.text_area(
            "Code to improve:",
            height=300,
            key="improve_input",
            placeholder="for i in range(len(list)):\n    print(list[i])"
        )
        
        if st.button("Get Suggestions", key="improve"):
            if code_for_improvement.strip():
                with st.spinner("Finding improvements..."):
                    suggestions = agent.suggest_improvements(code_for_improvement)
                    st.subheader("💡 Suggested Enhancements:")
                    for i, suggestion in enumerate(suggestions, 1):
                        st.write(f"{i}. {suggestion}")
            else:
                st.warning("Please enter code to analyze")
    
    # Tab 4: Line-by-Line Explanation
    with tab4:
        st.header("🔍 Line-by-Line Explanation")
        code_for_explanation = st.text_area(
            "Code to explain:",
            height=300,
            key="explain_input",
            placeholder="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        )
        
        if st.button("Explain Code", key="explain"):
            if code_for_explanation.strip():
                with st.spinner("Explaining code..."):
                    explanations = agent.explain_line_by_line(code_for_explanation)
                    st.subheader("📖 Line Explanations:")
                    for line_num, explanation in sorted(explanations.items()):
                        st.write(f"**Line {line_num}**: {explanation}")
            else:
                st.warning("Please enter code to explain")
    
    # Tab 5: Compare Implementations
    with tab5:
        st.header("⚖️ Compare Implementations")
        col1, col2 = st.columns(2)
        
        with col1:
            code1 = st.text_area(
                "Implementation 1:",
                height=300,
                key="compare1",
                placeholder="# Version 1"
            )
        
        with col2:
            code2 = st.text_area(
                "Implementation 2:",
                height=300,
                key="compare2",
                placeholder="# Version 2"
            )
        
        if st.button("Compare", key="compare"):
            if code1.strip() and code2.strip():
                with st.spinner("Comparing implementations..."):
                    comparison = agent.compare_implementations(code1, code2)
                    st.markdown(comparison)
            else:
                st.warning("Please enter code in both sections")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Code Explainer Agent** | Powered by Google Gemini API
    
    ✨ Features:
    - 🧠 AI-powered code analysis
    - 📝 Automatic documentation
    - 💡 Improvement suggestions
    - 🔍 Line-by-line explanations
    - ⚖️ Implementation comparisons
    """)

if __name__ == "__main__":
    main()
