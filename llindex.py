import os
from pathlib import Path
from typing import Dict, List, Optional

import openai

import llama_index
from llama_index import GPTVectorStoreIndex, download_loader
from llama_index.readers.schema.base import Document
from llama_index import SimpleWebPageReader, QuestionAnswerPrompt
from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document

class LL_INDEX:
    def __init__(self): 
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def _validate_prompt_template(self, prompt_template: str):
            if '{context_str}' not in prompt_template or '{query_str}' not in prompt_template:
                raise Exception("Provided prompt template is invalid, missing one of `{context_str}` or `{query_str}`. Please ensure both placeholders are present and try again.")  

    def fetch_openai_output(self, prompt):
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",  
                prompt=prompt,
                max_tokens=100  
            )
            return response['choices'][0]['text'].strip()
        except openai.error.AuthenticationError:
            return "Authentication error. Please check your API key."
        except openai.error.RateLimitError:
            return "Rate limit exceeded. Please wait a while and try again."
        except openai.error.APIError as e:
            return f"OpenAI API error: {e}"
    
    def load_pdf_data(self):
        """Parse file."""
        import pypdf

        import os
        import glob

        directory_path = '/Users/bseetharaman/Desktop/FY23/TPF_Hackathon/documents/hr'

        pdf_files = []
        for file in glob.glob(os.path.join(directory_path, '*.pdf')):
            pdf_files.append(file)
        
       
        docs = []
        print("List of PDF files in the directory:")
        for file in pdf_files:

            with open(file, "rb") as fp:
                pdf = pypdf.PdfReader(fp)
                num_pages = len(pdf.pages)
               
                for page in range(num_pages):
                    page_text = pdf.pages[page].extract_text()
                    page_label = pdf.page_labels[page]
                    metadata = {"page_label": page_label, "file_name": file}

                    docs.append(Document(text=page_text, extra_info=metadata))

        return docs
    
    def fetch_hr_qa_output(self, prompt):
        docs = self.load_pdf_data()
        index = GPTVectorStoreIndex.from_documents(docs)
        query_engine = index.as_query_engine()
        query_results = query_engine.query(prompt)
        return(query_results)
    
if __name__ == "__main__":
    llindex = LL_INDEX()
    prompt_text = "what is maximum amount for Educational & Learning Reimbursement?"
    output = llindex.fetch_hr_qa_output(prompt_text)
    
    # output = llindex.fetch_openai_output(prompt_text)
    print(output)

