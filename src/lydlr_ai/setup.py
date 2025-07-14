from setuptools import find_packages, setup

package_name = 'lydlr_ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'torch'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='AI Storage Optimization Package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = lydlr_ai.my_node:main',
            'optimizer_node = lydlr_ai.optimizer_node:main',
            'testing = lydlr_ai.test_publisher_node:main',
        ],
    },
)
