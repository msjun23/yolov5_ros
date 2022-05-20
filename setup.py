from setuptools import setup

package_name = 'yolov5_ros'
data = 'yolov5_ros/data'
models = 'yolov5_ros/models'
utils = 'yolov5_ros/utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, data, models, utils],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='msjun-ubuntu',
    maintainer_email='msjun23@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'Detector = yolov5_ros.Detector:main'
        ],
    },
)
