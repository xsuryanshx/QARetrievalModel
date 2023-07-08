import streamlit as st
from rag_qa_model import RAG_QA_Model
from pathlib import Path
import json
import pandas as pd
import os

st.set_page_config(page_title="QA Model", page_icon="🔎", layout="wide")


def initialize():
    if "engine" not in st.session_state:
        st.session_state.engine = RAG_QA_Model()
    if "vector_storage_type" not in st.session_state:
        st.session_state.vector_storage_type = "Normal"
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = "Input your API key"
    if "selected_document" not in st.session_state:
        st.session_state.selected_document = "Knowledge Document - Pan Card"
        load_documents()


def load_documents():
    with st.sidebar:
        with st.spinner(
            f"Loading {st.session_state.selected_document} into {st.session_state.vector_storage_type}..."
        ):
            st.session_state.engine.load_document(
                st.session_state.selected_document,
                st.session_state.vector_storage_type,
                st.session_state.openai_key,
            )
            st.session_state.total_pages_in_document = (
                st.session_state.engine.total_chunks
            )

def retrieve_documents():
    with open(Path("./document_config.json").resolve(), "r") as f:
        document_config = json.load(f)
        return tuple(document_config.keys())


@st.cache_data
def convert_df(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")


def process_question_as_text(
    engine: RAG_QA_Model,
    question: str,
    number_of_documents_to_review: int,
    temperature: float,
):
    st.write(
        f"You selected the following Document: **{st.session_state.selected_document}**"
    )
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

    st.write(f"Answer: \n {filtered_resulting_df['Answer'][0]}")

    show_df_as_table(
        filtered_resulting_df[
            [
                "Question",
                "Answer",
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

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.table(df.style.set_table_styles(styles))


def main():
    st.title("PAN Card Information Center")
    with st.sidebar:
        openai_api = st.text_input(
            "OpenAI API Key",
            key="openai_key",
            type="password",
            on_change=load_documents,
        )
        st.selectbox(
            "Select Document",
            retrieve_documents(),
            key="selected_document",
            on_change=load_documents,
        )
        st.selectbox(
            "Model Type:",
            ["Normal", "Multilingual"],
            key="vector_storage_type",
            on_change=load_documents,
            help="""The Normal Model uses ChromaDB vector storage and OpenAI embeddings which \
                is great for QA Retrieval in English Language. \n The Multilingual Model uses \
                Qdrant vector storage and Cohere embeddings which perform great for QA Retrieval \
                in Multiple Languages.""",
        )
        # "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"

    question_input = st.text_input("Enter your question", "")
    input_is_valid = question_input != ""

    if input_is_valid and question_input and not openai_api:
        st.info("Please add your OpenAI API key to continue.")

    col1, col2 = st.columns(2)
    with col1:
        number_of_documents_to_review = st.slider(
            "Number of Chunks to Consider",
            min_value=1,
            value=min(5, st.session_state.total_pages_in_document),
            step=1,
            max_value=st.session_state.total_pages_in_document,
        )
    with col2:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Run", disabled=(not input_is_valid), type="primary"):
        process_question_as_text(
            st.session_state.engine,
            question_input,
            number_of_documents_to_review,
            temperature,
        )


if __name__ == "__main__":
    initialize()
    main()
