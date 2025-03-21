from setuptools import setup, find_packages

setup(
    name="satellite-inferno-detector",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.31.0",
        "opencv-python-headless>=4.7.0.72",
        "torch>=2.0.0",
        "ultralytics>=8.1.0",
        "planetary-computer>=0.5.1",
        "pystac-client>=0.7.2",
        "pillow>=9.5.0",
        "pandas>=1.5.3",
        "numpy>=1.22.0",
    ],
    python_requires=">=3.8",
)
