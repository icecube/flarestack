import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flarestack",
    version="0.1",
    author="Robert Stein",
    author_email="robert.stein@desy.de",
    description="Unbinned likelihood analysis code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="astroparticle physics science unbinned likelihood IceCube",
    url="https://github.com/robertdstein/flarestack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy==1.14.0",
        "numexpr==2.6.4",
        "healpy==1.10.3",
        "scipy==1.1.0"
    ]
)