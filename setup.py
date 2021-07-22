from setuptools import find_packages, setup
setup(name="dmdeeg",
      version="0.1.0",
      description="Analyze EEG data with dynamic mode decomposition",
      author="Christian Goelz",
      author_email='c.goelz@gmx.de',
      platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
      license="MIT",
      url="http://github.com/christiangoelz/dmdeeg",
      packages=find_packages(),
      )