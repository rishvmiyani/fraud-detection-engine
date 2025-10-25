#!/usr/bin/env python3
"""
Fraud Detection Engine Setup
Legacy setup.py for backward compatibility
"""

from setuptools import setup, find_packages
import os
import sys

# Read version from pyproject.toml or __init__.py
def get_version():
    try:
        import tomli
        with open("pyproject.toml", "rb") as f:
            data = tomli.load(f)
        return data["project"]["version"]
    except ImportError:
        # Fallback to reading from __init__.py
        version_file = os.path.join("src", "__init__.py")
        if os.path.exists(version_file):
            with open(version_file, "r") as f:
                for line in f:
                    if line.startswith("__version__"):
                        return line.split("=")[1].strip().strip('"').strip("'")
        return "1.0.0"

# Read long description
def get_long_description():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def get_requirements(filename="requirements.txt"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

if __name__ == "__main__":
    setup(
        name="fraud-detection-engine",
        version=get_version(),
        author="Fraud Detection Team",
        author_email="team@frauddetection.com",
        description="Real-time Fraud Detection Engine with Machine Learning",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        url="https://github.com/fraud-detection/fraud-detection-engine",
        project_urls={
            "Bug Tracker": "https://github.com/fraud-detection/fraud-detection-engine/issues",
            "Documentation": "https://fraud-detection-engine.readthedocs.io/",
            "Source Code": "https://github.com/fraud-detection/fraud-detection-engine",
        },
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Financial and Insurance Industry",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Office/Business :: Financial",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Security",
        ],
        python_requires=">=3.10",
        install_requires=get_requirements("requirements.txt"),
        extras_require={
            "dev": get_requirements("requirements-dev.txt"),
            "test": get_requirements("requirements-test.txt"),
        },
        entry_points={
            "console_scripts": [
                "fraud-detection=src.cli:main",
            ],
        },
        include_package_data=True,
        zip_safe=False,
    )
