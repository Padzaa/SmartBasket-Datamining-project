from setuptools import setup, find_packages

setup(
    name="market_basket_analytics",
    version="1.0.0",
    description="State-of-the-art Market Basket Analysis with Multi-Industry Applications",
    author="Victor Collins Oppon",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "mlxtend>=0.22.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "streamlit>=1.28.0",
        "networkx>=3.1",
    ],
    python_requires=">=3.8",
)
