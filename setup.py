from setuptools import setup, find_packages

setup(
    name='davis_stats',
    version='1.0',
    packages=find_packages(),
    package_data={
        'davis_stats': ['data/*.xlsx']},
    install_requires=[
        'pandas>=1.0.0',
        'openpyxl>=3.0.0',
        'numpy>=1.20.0',
        'matplotlib>=3.0.0',
        'scipy>=1.6.0',
        'seaborn>=0.12.0',
        'statsmodels>=0.14.0',
        'ipympl>=0.8.0'],
    python_requires='>=3.7',
    author='Justin G. Davis',
    author_email='',
    description='''
        davis_stats is a teaching-focused repository 
        for applied statistics, data science, and 
        business analytics. It contains functions 
        and datasets to help W&L students develop 
        practical skills in data analysis, statistical 
        modeling, and real-world decision making.''',
    keywords='statistics, data science, business analytics')
