from setuptools import setup, find_packages

setup(
    name="unified_indexer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "pdf": ["pdfplumber>=0.7.0"],
        "docx": ["python-docx>=0.8.11"],
        "all": [
            "pdfplumber>=0.7.0",
            "python-docx>=0.8.11",
            "beautifulsoup4>=4.11.0",
        ],
    },
    python_requires=">=3.8",
)
