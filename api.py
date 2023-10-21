from similarities import *
from fastapi import FastAPI, UploadFile
import uvicorn
from langchain.document_loaders import PyPDFLoader

app = FastAPI()


@app.post("/get_similarity")
def main(doc1: UploadFile, doc2: UploadFile, distance_to_calculate: str):
    file1 = PyPDFLoader(doc1)
    file2 = PyPDFLoader(doc2)
    file1, file2 = file1.load(), file2.load()
    if distance_to_calculate == "cosine":
        output = cosine_similarity_(file1, file2)
        if output == 1.0:
            return "Identical document"
        elif output == 0.0:
            return "Non Identical document"
        return output
    elif distance_to_calculate == "bleu":
        output = get_bleu_score(file1, file2)
        if output == 1.0:
            return "Identical document"
        elif output == 0.0:
            return "Non Identical document"
        return output
    elif distance_to_calculate == "levenshtein_distance":
        output = get_levenshtein_similarity(file1, file2)
        if output == 1.0:
            return "Identical document"
        elif output == 0.0:
            return "Non Identical document"
        return output
    elif distance_to_calculate == "jaccard":
        output = get_jaccard_similarity(file1, file2)
        if output == 1.0:
            return "Identical document"
        elif output == 0.0:
            return "Non Identical document"
        return output
    elif distance_to_calculate == "eucleudian":
        output = get_euclidean_similarity(file1, file2)
        if output == 0.0:
            return "Identical document"
        elif output == 1.0:
            return "Non Identical document"
        return output
    elif distance_to_calculate == "dice_coefficient":
        output = get_dice_coefficient(file1, file2)
        if output == 1.0:
            return "Identical document"
        elif output == 0.0:
            return "Non Identical document"
        return output
    elif distance_to_calculate == "wer":
        output = get_word_error_rate(file1, file2)
        if output == 0.0:
            return "Identical document"
        elif output == 1.0:
            return "Non Identical document"
        return output


# reference_document = "Mlops Course Curriculum.pdf"
# candidate_document = "Mlops Course Curriculum.pdf"

# print(main(reference_document, candidate_document, "wer"))

if __name__ == "__main__":
    uvicorn.run(app=app, port=8000, host="specify")
