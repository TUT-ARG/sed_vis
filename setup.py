from setuptools import setup

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')

except (IOError, ImportError):
    long_description = open('README.md').read()

setup(
    name='sed_vis',
    version='0.1.2',
    description='Visualization tools for sound event detection research.',
    author='Toni Heittola',
    author_email='toni.heittola@gmail.com',
    url='https://github.com/TUT-ARG/sed_vis/',
    packages=['sed_vis'],
    long_description=long_description,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        'Development Status :: 5 - Production/Stable',
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2.7",
    ],
    keywords=['audio analysis', 'sound event detection', 'dsp'],
    license='MIT',
    install_requires=[
        'numpy >= 1.7.0',
        'scipy >= 0.9.0',
        'matplotlib >= 1.4.0',
        'pyaudio >= 0.2.7',
    ],
)
