from setuptools import find_packages, setup

package_name = 'urubots_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/robocup.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vinicio',
    maintainer_email='vinicio.melgar@estudiantes.utec.edu.uy',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'robocup_mapper = urubots_vision.robocup_mapper:main',
            'vision_detector = urubots_vision.vision_detector:main',
            'geotiff_mapper = urubots_vision.geotiff_mapper:main'
        ],
    },
)
