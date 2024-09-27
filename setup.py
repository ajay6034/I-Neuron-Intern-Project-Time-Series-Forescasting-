from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads the requirements from the file
    and returns a list of package names.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove newline characters and ignore empty lines and comments
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

        # Remove '-e .' for editable install if it exists
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Ajay',
    author_email='ajaykumarjagu155@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
