import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flarestack",
    version="2.4.5",
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "astropy==5.1",
        "black==22.6.0",
        "coveralls==3.3.1",
        "healpy==1.16.1",
        "jupyter==1.0.0",
        "matplotlib==3.5.2",
        "numexpr==2.8.3",
        "numpy==1.23.1",
        "scipy==1.9.0",
        "setuptools==63.4.1",
        "sphinx==5.1.1",
    ],
    package_data={
        "flarestack": [
            "data/public/icecube/all_sky_point_source/raw_data/3year-data-release.zip"
        ]
    },
    include_package_data=True,
)
