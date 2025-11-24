from setuptools import setup

setup(
    name='peptide-pose-predictor',
    version='1.0.0',
    description='Peptide Pose Predictor using EGNN',
    packages=['peptide_pose_predictor'],
    package_data={'peptide_pose_predictor': ['best_model_egnn.pth']},
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
            'ppp=peptide_pose_predictor.predict:main',
        ],
    },
)
