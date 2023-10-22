from similarities import *
import streamlit as st
import os
import shutil
from langchain.document_loaders import PyPDFLoader


def main(doc1: str, doc2: str, distance_to_calculate: str):
    if doc1.split(".")[-1] == "pdf" and doc2.split(".")[-1] == "pdf":
        file1 = PyPDFLoader(doc1)
        file2 = PyPDFLoader(doc2)
        file1_, file2_ = file1.load(), file2.load()
        file1 = "".join([doc.page_content for doc in file1_])
        file2 = "".join([doc.page_content for doc in file2_])
    elif doc1.split(".")[-1] == "txt" and doc2.split(".")[-1] == "pdf":
        file1 = open(doc1, "r", encoding="utf-8").read()
        file2 = PyPDFLoader(doc2)
        file2_ = file2.load()
        file2 = "".join([doc.page_content for doc in file2_])
    elif doc1.split(".")[-1] == "pdf" and doc2.split(".")[-1] == "txt":
        file2 = open(doc2, "r", encoding="utf-8").read()
        file1 = PyPDFLoader(doc1)
        file1_ = file1.load()
        file1 = "".join([doc.page_content for doc in file1_])
    else:
        file1 = open(doc1, "r", encoding="utf-8").read()
        file2 = open(doc2, "r", encoding="utf-8").read()
    if distance_to_calculate == "cosine":
        output = cosine_similarity_(file1, file2)
        if output == 1.0:
            return f"cosine: Identical document,Score: {output}"
        elif output == 0.0:
            return f"cosine: Non Identical document,Score: {output}"
        return f"cosine: {output}"
    elif distance_to_calculate == "bleu":
        output = get_bleu_score(file1, file2)
        if output == 1.0:
            return f"bleu: Identical document,Score: {output}"
        elif output == 0.0:
            return f"bleu: Non Identical document,Score: {output}"
        return f"bleu: {output}"
    elif distance_to_calculate == "levenshtein_distance":
        output = get_levenshtein_similarity(file1, file2)
        if output == 1.0:
            return f"levenshtein_distance: Identical document,Score: {output}"
        elif output == 0.0:
            return f"levenshtein_distance: Non Identical document,Score: {output}"
        return f"levenshtein_distance: {output}"
    elif distance_to_calculate == "jaccard":
        output = get_jaccard_similarity(file1, file2)
        if output == 1.0:
            return f"jaccard: Identical document,Score: {output}"
        elif output == 0.0:
            return f"jaccard: Non Identical document,Score: {output}"
        return output
    elif distance_to_calculate == "eucleudian":
        output = get_euclidean_similarity(file1, file2)
        if output == 0.0:
            return f"eucleudian: Identical document,Score: {output}"
        elif output == 1.0:
            return f"eucleudian: Non Identical document,Score: {output}"
        return output
    elif distance_to_calculate == "dice_coefficient":
        output = get_dice_coefficient(file1, file2)
        if output == 1.0:
            return f"dice_coefficient: Identical document,Score: {output}"
        elif output == 0.0:
            return f"dice_coefficient: Non Identical document,Score: {output}"
        return f"dice_coefficient: {output}"
    elif distance_to_calculate == "wer":
        output = get_word_error_rate(file1, file2)
        if output == 0.0:
            return f"wer: Identical document,Score: {output}"
        elif output == 1.0:
            return f"wer: Non Identical document,Score: {output}"
        return f"wer: {output}"


def streamlit_main():
    os.makedirs("uploads", exist_ok=True)

    st.title("Document Similarity Checker")

    st.sidebar.header("Upload Documents")
    doc1 = st.sidebar.file_uploader("Upload First Document", type=["pdf", "txt"])
    doc2 = st.sidebar.file_uploader("Upload Second Document", type=["pdf", "txt"])
    distance_to_calculate = st.sidebar.selectbox(
        "Select Distance Calculation Method",
        [
            "cosine",
            "bleu",
            "levenshtein_distance",
            "jaccard",
            "eucleudian",
            "dice_coefficient",
            "wer",
        ],
    )

    if st.sidebar.button("Submit"):
        if doc1 is not None and doc2 is not None:
            with open(os.path.join("uploads", doc1.name), "wb") as f:
                f.write(doc1.read())
            with open(os.path.join("uploads", doc2.name), "wb") as f:
                f.write(doc2.read())

            # Get file paths
            doc1_path = os.path.join("uploads", doc1.name)
            doc2_path = os.path.join("uploads", doc2.name)

            result = main(doc1_path, doc2_path, distance_to_calculate)
            st.subheader("Result")
            st.write(result)
            shutil.rmtree("uploads")


if __name__ == "__main__":
    streamlit_main()

# reference_document = "Mlops Course Curriculum.pdf"
# candidate_document = "Mlops Course Curriculum.pdf"

# print(main(reference_document, candidate_document, "wer"))
