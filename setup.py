from setuptools import setup, find_packages

setup(
    name='lerobot',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add dependencies here, e.g., 'requests>=2.25.1'
    ],
    authors = [
        "Rémi Cadène <re.cadene@gmail.com>",
        "Simon Alibert <alibert.sim@gmail.com>",
        "Alexander Soare <alexander.soare159@gmail.com>",
        "Quentin Gallouédec <quentin.gallouedec@ec-lyon.fr>",
        "Adil Zouitine <adilzouitinegm@gmail.com>",
        "Thomas Wolf <thomaswolfcontact@gmail.com>",
    ],
    description='A poetry project for sharing poetry',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/huggingface/lerobot",  # Your project's URL
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        # "Programming Language :: Python :: 3.10",
    ],
    python_requires='>=3.8',  # Set minimum Python version
)