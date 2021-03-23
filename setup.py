from setuptools import find_packages, setup

# extract version from __init__.py
with open('socialforce/__init__.py', 'r') as f:
    VERSION_LINE = [l for l in f if l.startswith('__version__')][0]
    VERSION = VERSION_LINE.split('=')[1].strip()[1:-1]


setup(
    name='socialforce',
    version=VERSION,
    packages=find_packages(),
    license='AGPL',
    description='PyTorch implementation of DeepSocialForce.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sven Kreiss',
    author_email='research@svenkreiss.com',
    url='https://github.com/svenkreiss/socialforce',

    install_requires=[
        'numpy',
        'torch',
    ],
    extras_require={
        'dev': [
            'nbstripout',
            'pylint',
            'pytest',
        ],
        'plot': [
            'flameprof',
            'matplotlib',
            'scipy',
        ],
    },

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: PyPy',
    ]
)
