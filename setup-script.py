"""
Peso Project - Setup script
"""

from setuptools import setup, find_packages

setup(
    name="peso",
    version="0.1.0",
    description="Non-human marketing data warehouse project",
    author="Peso Project Team",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "requests>=2.26.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Marketing Professionals",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)
