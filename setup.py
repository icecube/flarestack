import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flarestack",
    version="2.4.0",
    author="Robert Stein",
    author_email="robert.stein@desy.de",
    description="Package for unbinned likelihood analysis of physics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="astroparticle physics science unbinned likelihood IceCube",
    url="https://github.com/icecube/flarestack",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires='>=3.7',
    install_requires=[
        "setuptools==57.0.0",
        "numpy==1.21.0",
        "numexpr==2.7.3",
        "healpy==1.15.0",
        "scipy==1.7.0",
        "matplotlib==3.4.2",
        "astropy==4.2.1",
        "sphinx==4.0.2",
        "jupyter==1.0.0",
        "coveralls==3.1.0"
    ],
    package_data={'flarestack': [
        'data/public/icecube/all_sky_point_source/raw_data/3year-data-release.zip']},
    include_package_data=True
)

