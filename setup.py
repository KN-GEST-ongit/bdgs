from setuptools import setup, find_packages

setup(
    name="bdgs",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'opencv-python~=4.11.0.86',
        'numpy~=2.1.3',
        'tensorflow-cpu~=2.19.0',
        'scikit-learn~=1.6.1',
        "keras~=3.9.2",
        'scikit-image~=0.25.2',
        'silence-tensorflow~=1.2.3'
    ],
    include_package_data=True,
    package_data={
        'bdgs_trained_models': ['*'],
    },
    py_modules=['definitions'],
    author="GEST science club, Rzeszów University of Technology",
    author_email="kn.gest@kia.prz.edu.pl",
    description="Static gestures recognition tool",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
