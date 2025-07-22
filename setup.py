from setuptools import find_packages,setup

def installrequires(filepath):
    requirements = []
    with open(filepath,'r') as f:
        requirements = f.readlines()
        requirements = [req.replace('\n','') for req in requirements]

    if '-e .' in requirements:
        requirements.remove('-e .')

    return requirements




setup(
    name='Medical_Chatbot',
    version='0.1.0',
    author='Sahil Himmatbhai Katariya',
    author_email='sahilkatariya012@gmail.com',
    packages=find_packages(),
    install_requires = installrequires('requirements.txt')
)