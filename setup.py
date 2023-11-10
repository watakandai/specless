from setuptools import setup, find_packages
import pathlib
import pkg_resources

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='specless',
    version='0.1.0',
    author='Kandai Watanabe',
    author_email='kandai.wata@gmail.com',
    long_description=readme,
    url='https://github.com/watakandai/specless',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=install_requires,
    python_requires='>=3.7',
)
