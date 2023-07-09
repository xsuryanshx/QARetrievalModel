# QARetrieval Model 
AI Question Answering model that answers the important questions related to PAN card. 

![image](https://github.com/xsuryanshx/QARetrievalModel/assets/51471876/0e296aea-074e-454b-a1f6-802bafab858e)


## Features
Incorporated multiple features in this update including :
*  Multilinguality -  for asking questions in different languages
*  Speech - Use speech to ask questions and return answer in speech.
*  Scores -  Generated Rouge Score.
   
## Running Locally
Follow these steps to set up and run the service locally :

### Prerequisites
- Python 3.8 or higher
- Git

### Installation
Clone the repository :

`https://github.com/xsuryanshx/QARetrievalModel.git`

Navigate to the project directory :

`cd QARetrievalModel`


Create a virtual environment :
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Install the required dependencies in the virtual environment :

`pip install -r requirements.txt`

`pip install -r packages.txt`

Launch the chat service locally :

`streamlit run streamlit_qa.py`

#### That's it! The service is now up and running locally. 🤗


