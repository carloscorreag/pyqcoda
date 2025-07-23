from setuptools import setup, find_packages

setup(
    name="pyqcoda",
    version="1.0.0",
    author="Carlos Correa Guinea",
    author_email="ccorreag@aemet.es",
    description="Temporal disaggregation of daily precipitation into hourly using Q-CODA.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/pyqcoda",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires='>=3.7',
    include_package_data=True,
)

