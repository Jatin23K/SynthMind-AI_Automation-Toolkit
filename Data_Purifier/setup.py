from setuptools import setup, find_packages

setup(
    name="data_purifier",
    version="0.1.0",
    description="A data cleaning and transformation toolkit",
    author="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "ydata-profiling>=4.5.0",
        "crewai>=0.1.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0"
    ],
)
