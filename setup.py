from setuptools import setup, find_packages

setup(
    name="code_explainer_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'requests>=2.25.1',
        'pydantic>=1.10.0',
        'python-dotenv>=0.19.0',
        'astor>=0.8.1',
        'astunparse>=1.6.3',
        'sentence-transformers>=2.2.2',
        'transformers>=4.30.0',
        'torch>=2.0.0',
        'streamlit>=1.24.0',
        'fastapi>=0.95.0',
        'uvicorn>=0.21.0',
        'python-multipart>=0.0.6',
        'langgraph>=0.0.10',
        'langchain>=0.0.350',
    ],
    python_requires='>=3.8',
)
