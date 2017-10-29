from setuptools import find_packages, setup

setup(
    name='classifier',
    packages=['classifier'],
    package_data={'':['iris.csv']},
    zip_safe=False
)
