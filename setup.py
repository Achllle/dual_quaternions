from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='dual_quaternions',
      version='0.3.4',
      description='Dual quaternion implementation',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/Achllle/dual_quaternions',
      author='Achille Verheye',
      author_email='achille.verheye@gmail.com',
      license='MIT',
      packages=['dual_quaternions'],
      package_dir={'': 'src'},
      install_requires=['numpy', 'pyquaternion'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require='nose')
