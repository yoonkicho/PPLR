from setuptools import setup, find_packages


setup(name='PPLR',
      version='1.0.0',
      description='Part-based Pseudo Label Refinement for Unsupervised Person Re-identification',
      author='Yoonki Cho',
      author_email='yoonki@kaist.ac.kr',
      url='https://github.com/yoonkicho/PPLR',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu==1.6.3'],
      packages=find_packages()
      )