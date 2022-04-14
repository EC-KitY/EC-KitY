from setuptools import setup, find_packages
import eckity

VERSION = eckity.__version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='eckity',
    version=VERSION,
    author='Moshe Sipper',
    author_email='sipper@gmail.com',
    description='EC-KitY: Evolutionary Computation Tool Kit in Python.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/moshesipper/ec-kity',
    project_urls = {
        "Bug Tracker": "https://github.com/moshesipper/ec-kity/issues"
    },
    license='GNU GPLv3',
    packages=find_packages(),
    install_requires=['scikit-learn>=0.24.2'],
)
