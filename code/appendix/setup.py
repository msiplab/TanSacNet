from setuptools import setup, find_packages

setup(
    name='torch-tansacnet',
    version='0.1',
    packages=find_packages(),
    python_requires='>=3.10',    
    install_requires=[
        'torch>=2.3.0',
        'torchvision>=0.18.0',
        'torch-dct>=0.1.6',
        'scipy>=1.8.1',
        'parameterized>=0.9.0',
    ],
    author='Shogo MURAMATSU',
    author_email='shogo@eng.niigata-u.ac.jp',
    description='TANSACNet: Tangent Space Adaptive Control Networks',
    url='https://github.com/msiplab/TANSACNet'
)
