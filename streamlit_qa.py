import streamlit as st
from rag_qa_model import RAG_QA_Model
from speech_recog import speech_recognition
from pathlib import Path
import json
import pandas as pd
import os
from io import BytesIO
from gtts import gTTS


st.set_page_config(page_title="QA Model", page_icon="🔎", layout="wide")


def initialize():
    """initializer funcion"""
    if "engine" not in st.session_state:
        st.session_state.engine = RAG_QA_Model()
    if "vector_storage_type" not in st.session_state:
        st.session_state.vector_storage_type = "Normal"
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = ""
    if "selected_document" not in st.session_state:
        st.session_state.selected_document = "Knowledge Document - Pan Card"
        load_documents()


def load_documents():
    """loader for session state of documents"""
    with st.sidebar:
        with st.spinner("Loading Model ..."):
            st.session_state.engine.load_document(
                st.session_state.selected_document,
                st.session_state.vector_storage_type,
                st.session_state.openai_key,
            )
            st.session_state.total_pages_in_document = (
                st.session_state.engine.total_chunks
            )


def retrieve_documents():
    """returns a list of documents in json

    Returns:
        tuple: list of documents
    """
    with open(Path("./document_config.json").resolve(), "r") as f:
        document_config = json.load(f)
        return tuple(document_config.keys())


@st.cache_data
def convert_df(df: pd.DataFrame):
    """converts df into csv, seperate function to delete cache stored if needed.
    Args:
        df: inputs dataframe
    Returns:
        csv: returns csv file
    """
    return df.to_csv(index=False).encode("utf-8")


def process_question_as_text(
    engine: RAG_QA_Model,
    question: str,
    number_of_documents_to_review: int,
    temperature: float,
):
    """Takes input question and other parameters to processes the answers using GPT.

    Args:
        engine (RAG_QA_Model): model file
        question (str): string
        number_of_documents_to_review (int): number of chunks of document/text we want to use
        temperature (float): temperature to control
    """
    st.write("-----------------------------------------------------------")
    with st.spinner("Processing using GPT..."):
        resulting_df = engine.answer_questions(
            question,
            number_of_documents_to_review,
            temperature,
        )
    resulting_csv = convert_df(resulting_df)
    filtered_resulting_df = resulting_df.copy(deep=True)
    st.write(
        f"This request took approximately **{filtered_resulting_df['Request Time (s)'][0]} seconds**"
    )

    output_answer = filtered_resulting_df["Answer"][0]

    st.write(f"Answer: \n {output_answer}")
    with st.spinner("Generating speech to text output...."):
        speech_output(output=output_answer)
    with st.spinner("Loading answer dataframe...."):
        show_df_as_table(
            filtered_resulting_df[
                [
                    "Question",
                    "Answer",
                    "Score",
                    "Request Time (s)",
                    "Total Cost ($)",
                    "Total Tokens",
                ]
            ]
        )
    st.download_button(
        "Download the Results",
        resulting_csv,
        "results.csv",
        "text/csv",
        key="download-csv",
    )


def show_df_as_table(df: pd.DataFrame):
    """Code to show dataframe as table in streamlit"""
    th_props = [
        ("text-align", "center"),
        ("font-weight", "bold"),
    ]

    td_props = [("text-align", "left"), ("font-size", "14px")]

    styles = [dict(selector="th", props=th_props), dict(selector="td", props=td_props)]

    hide_table_row_index = """
    <style>
        thead tr th:first-child {display:none}
        tbody th {display:none}
    </style>
    """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.table(df.style.set_table_styles(styles))


def speech_output(output):
    """Returns a audio response of the answer.
    Args:
        output (str): string
    """
    sound = BytesIO()
    tts = gTTS(output, lang="en", tld="com")
    tts.write_to_fp(sound)
    c, _ = st.columns(2)
    with c:
        st.audio(sound)


def main():
    """main function"""
    st.title("PAN Card Information Center")
    with st.sidebar:
        openai_api = st.text_input(
            "OpenAI API Key",
            key="openai_key",
            type="password",
            on_change=load_documents,
        )
        st.write(
            "Note: Please input your OpenAI key. If you don't have that then please\
            limit to use the platform to 2-3 tries as it uses my current API key."
        )
        st.selectbox(
            "Select Document",
            retrieve_documents(),
            key="selected_document",
            on_change=load_documents,
        )
        st.radio(
            "Model Type:",
            ("Normal", "Multilingual"),
            key="vector_storage_type",
            on_change=load_documents,
            help="""The Normal Model uses ChromaDB vector storage and OpenAI embeddings which \
                is great for QA Retrieval in English Language. \n The Multilingual Model uses \
                Qdrant vector storage and Cohere embeddings which perform great for QA Retrieval \
                in Multiple Languages.""",
        )
        "[View the source code](https://github.com/xsuryanshx/QARetrievalModel/tree/develop)"
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/xsuryanshx/QARetrievalModel/tree/develop?quickstart=1)"

    question_input = st.text_input("Enter your question", "")
    input_is_valid = question_input != ""

    if input_is_valid and question_input and not openai_api:
        st.info("Please add your OpenAI API key to continue.")
    else:
        load_documents()

    col1, col2, _ = st.columns([1, 1, 8])
    with col1:
        run_button = st.button("Run", disabled=(not input_is_valid), type="primary")
    with col2:
        speak_button = st.button("Speak", type="primary")

    col3, col4 = st.columns(2)
    with col3:
        number_of_documents_to_review = st.slider(
            "Number of Chunks of text to use",
            min_value=1,
            value=min(5, st.session_state.total_pages_in_document),
            step=1,
            max_value=st.session_state.total_pages_in_document,
        )
    with col4:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, step=0.01)

    if run_button:
        process_question_as_text(
            st.session_state.engine,
            question_input,
            number_of_documents_to_review,
            temperature,
        )

    if speak_button:
        with st.spinner("Listening...."):
            question_input = st.text_input("Asked question", f"{speech_recognition()}?")
            process_question_as_text(
                st.session_state.engine,
                question_input,
                number_of_documents_to_review,
                temperature,
            )


if __name__ == "__main__":
    initialize()
    main()
