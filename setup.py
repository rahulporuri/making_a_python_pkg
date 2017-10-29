from setuptools import find_packages, setup

setup(
    name='classifier',
    packages=['classifier'],
    package_data={'':['iris.csv']},
    zip_safe=False,
    install_requires=['scikit_learn', 'scipy', 'matplotlib'],
    entry_points={
        'console_scripts': [
            'classify = classifier.plot_iris_dataset:main'
        ]
    }
)
