from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'tensor2tensor'
]

setup(
    name='open_web_text',
    version='0.1',
    author = 'Zhiyu',
    author_email = 'edward.liang@mail.utoronto.ca',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='OpenWebText LM Problem',
    requires=[]
)
