from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import json
import os
from langchain_community.document_loaders import PyPDFLoader
from datetime import datetime
from tqdm import tqdm
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from pprint import pprint

documents_directory = "./documents"
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
prompt_template = """
You are a text publication date finder. Analyze the text at the end to find the publication date.
If there is no publication date in it simply return no publication date.
Also differentiate between publication dates and other dates.
It is better if you find no date than a wrong publication date!
Think step by step.
Always output the date in the format YYYY/mm/dd
Your outputs should look like these json examples:
----------------------------------------
OUTPUT SAMPLES:
----------------------------------------
{{ 
    "success": true,
    "publication_date": "2021/01/01"
}}
{{ 
    "success": false,
    "publication_date": "N/A"
}}
----------------------------------------
TEXT TO ANALYZE:
----------------------------------------
{document}
----------------------------------------
"""


prompt = PromptTemplate(input_variables=["document"], template=prompt_template)
OPEN_AI_API_KEY = ""
llm = ChatOpenAI(temperature=0, openai_api_key=OPEN_AI_API_KEY)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
)


def split_pdf_pages(file_path):
    loader = PyPDFLoader(file_path)
    return (loader.load_and_split(), loader.load_and_split(splitter))


def get_publication_dates(filenames):
    final_result = []
    docs = []
    for file in tqdm(filenames):
        splitted_by_page_docs, splitted_docs = split_pdf_pages(file)
        for i, page in enumerate(splitted_by_page_docs):
            result = chain.invoke({"document": page.page_content})
            resultJSON = json.loads(result["text"])
            if resultJSON["success"] and resultJSON["publication_date"] != "N/A":
                date_string = resultJSON["publication_date"]
                date_object = datetime.strptime(date_string, "%Y/%m/%d")
                final_result.append(
                    {
                        "file_name": file,
                        "publication_date": date_object,
                    }
                )
                break
            if i == len(splitted_by_page_docs) - 1:
                final_result.append({"file_name": file, "publication_date": "N/A"})

        for doc in splitted_docs:
            doc.metadata["file_name"] = file
            doc.metadata["publication_date"] = date_object
        docs += splitted_docs
    return final_result, docs


def get_all_filenames(directory):
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]


if __name__ == "__main__":
    filenames = get_all_filenames(documents_directory)
    result, docs = get_publication_dates(filenames)
    print("FINAL RESULT")
    not_working = 0
    for res in result:
        print(
            f"File Name: {res['file_name']}, Publication Date: {res['publication_date']}"
        )
        if res["publication_date"] == "N/A":
            not_working = not_working + 1
    print("Number of not working: " + str(not_working))
