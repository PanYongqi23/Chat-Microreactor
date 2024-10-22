# ChatReactor Project Deployment Guide
Welcome to the ChatReactor project! This document provides a step-by-step guide to help you install and run the project smoothly.
## Environment Setup
### 1. Create a Conda Environment
To begin, you need to create a Conda environment named "ChatReactor". Open your command line tool and execute the following command:
```bash
conda create -n ChatReactor python=3.10
```
### 2. Activate the Environment
Once the environment is created, activate the "ChatReactor" environment with the following command:
```bash
conda activate ChatReactor
```
### 3. Install Dependencies
Next, install the dependencies required for the project. An `environment.yml` file is provided with the project. Use the following command to install the dependencies:
```bash
pip install -r requirements.txt
```
## Configuration
### 4. Configure OpenAI API
Locate the `data/accounts.txt` file in the project directory. Open it with a text editor and enter your OpenAI API key in the following format:
```
sk-wewqeqrsda1232dsadsdsads
```
Replace the example above with your actual API key.
## Running the Project
### 5. Run the Main Program
After completing the steps above, you can run the main program of the project. Navigate to the `src` subdirectory within the project folder and execute the following command:
```bash
streamlit run .\main.py
```
Upon successful execution, your web browser should open automatically, displaying the ChatReactor interface.
## Common Issues
- If you encounter issues with dependency installation, ensure that your Conda environment is correctly set up and that your internet connection is stable.
- If errors occur when running the main program, check that the API key in the `data/accounts.txt` file is entered correctly.
Thank you for using the ChatReactor project. If you have any questions or suggestions, please feel free to reach out to us. Enjoy your experience with ChatReactor!