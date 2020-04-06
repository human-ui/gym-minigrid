from setuptools import setup

setup(
    name='gym_minigrid',
    version='2.0',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    url='https://github.com/maximecb/gym-minigrid',
    description='Minimalistic gridworld package for OpenAI Gym with multiple environments running in paraller',
    packages=['gym_minigrid'],
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0',
        'sty'
    ]
)
