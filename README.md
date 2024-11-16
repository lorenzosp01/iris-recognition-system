# Iris Recognition System

## Table of Contents
- [Introduction](#introduction)
- [Verification flow](#verification-flow)
- [Dependencies](#dependencies)
- [Usage](#usage)


## Introduction
This project is an iris recognition system that uses the Daugman's Rubber Sheet Model to extract the iris features. 
The system is implemented in Python and uses OpenCV for image processing.

## Verification flow
![Alt text](images/verification_flow.png "Verification flow")

## Dependencies
The project dependencies are listed in the requirements.txt file.
To install the dependencies, run the following command:
```
pip install -r requirements.txt
```
or
```
pip-sync
```

To add file to package dependencies, run the following command:
```
pip freeze > requirements.txt
```

## Usage
To run the system, execute the following command:
```
python iris_recognition.py
```

