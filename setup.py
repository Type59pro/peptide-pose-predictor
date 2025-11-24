from setuptools import setup

setup(
    name='peptide-pose-predictor',
    version='1.0.0',
    description='Peptide Pose Predictor using EGNN',
    py_modules=['predict', 'build_graph', 'inference', 'training'],
    install_requires=[
        'numpy',
        'biopython',
        'networkx',
        'torch',
        'torch_geometric',
        'tqdm',
        'pandas'
    ],
    entry_points={
        'console_scripts': [
            'ppp=predict:main',
        ],
    },
)
