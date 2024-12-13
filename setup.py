from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]


setup(
    name="clinical-history-eval",
    version=0.1,
    description="Assessing the Completeness of Clinical Histories Accompanying Imaging Orders using Open- and Closed-Source Large Language Models",
    author="Arogya Koirala",
    author_email="arogya@stanford.edu",
    packages=find_namespace_packages(),
    python_requires=">=3.10",
    install_requires=[required_packages],
)