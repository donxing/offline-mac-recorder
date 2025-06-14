from setuptools import setup, find_packages

setup(
    name="offline-mac-recorder",
    version="0.1.0",
    description="Offline Voice Recorder and Transcriber for macOS",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="donxing",
    author_email="your.email@example.com",
    url="https://github.com/donxing/offline-mac-recorder",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["config.ini", "*.py", "scripts/*.sh"],
    },
    install_requires=open("requirements.txt").readlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "start-recorder=offline_mac_recorder.realtime_transcriber_llm:main",
            "start-api=offline_mac_recorder.main:main",
        ],
    },
)