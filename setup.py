from setuptools import setup, find_packages

setup(
    name="ppe-safety-compliance-monitoring",
    version="1.0.0",
    description="Real-time PPE Detection and Safety Compliance Monitoring System",
    author="SafetyGuard AI",
    packages=find_packages(),
    install_requires=[
        'ultralytics>=8.0.0',
        'huggingface_hub>=0.20.0',
        'opencv-python>=4.8.0',
        'streamlit>=1.28.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'pillow>=10.0.0',
        'pyyaml>=6.0',
    ],
    python_requires='>=3.8',
)
