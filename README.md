# Code Explainer Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/Gemini-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)

**Your AI-powered code analysis companion** - Understand, document, and improve your code with the power of Google's Gemini AI.

<div align="center">
  <img src="https://img.icons8.com/color/96/000000/code.png" alt="Code Explainer Logo" width="100"/>
  <p>Transform complex codebases into well-documented, maintainable projects</p>
</div>

## Features

- **AI-Powered Analysis**: Leverage Google's Gemini AI for intelligent code explanations
- **Multi-Language Support**: Works with Python, JavaScript, TypeScript, and Java
- **Interactive Web Interface**: Beautiful Streamlit-based UI for easy interaction
- **Code Metrics**: Get instant insights into code complexity and quality
- **File Upload**: Analyze entire code files with a simple drag-and-drop
- **Safety-First**: Built with robust safety filters for responsible AI usage
- **Educational Focus**: Designed to help developers learn and understand code better

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/pranav271103/Code-Explainer-Agent.git
   cd Code-Explainer-Agent
   ```

2. **Set up a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**
   Create a `.env` file in the project root and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

## Quick Start

1. **Run the Streamlit app**:
   ```bash
   streamlit run code_explainer_agent/app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Paste your code** or upload a file and click "Analyze Code"

## Features in Detail

### AI-Powered Code Analysis
Get detailed explanations of what your code does, how it works, and potential improvements - all powered by Google's Gemini AI.

### Code Metrics
- Cyclomatic complexity analysis
- Function length tracking
- Parameter count validation
- Code smell detection

### File Support
- Direct code input
- File upload (.py, .txt)
- Multiple files in a directory (coming soon)

### Safety & Privacy
- Local processing of code
- Configurable safety filters
- No data is stored or shared

## Configuration

Customize the agent's behavior by modifying `config.py`:

```python
class AgentConfig:
    llm_provider = "gemini"  # Currently supports Gemini
    llm_model = "gemini-2.5-flash"  # Model to use
    temperature = 0.2  # Lower for more deterministic output
    enable_memory = False  # Experimental: Enable conversation memory
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Resources

- [Google Gemini API Documentation](https://ai.google.dev/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Python Official Documentation](https://docs.python.org/3/)

---

<div align="center">
  Made by Pranav | 
  <a href="https://github.com/pranav271103">GitHub</a>
</div>
