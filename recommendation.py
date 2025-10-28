# importing necessary libraries
import ast
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate     
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser  
from langchain_huggingface import HuggingFaceEmbeddings     
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

# loading dataset
movies = pd.read_csv("data/raw/movies.csv", usecols=['genres', 'title', 'overview'])
movies.dropna(inplace=True)

# Preprocessing - convertin JSON like structure into normal format
def convert(text):
    ext = []
    for i in ast.literal_eval(text):
        ext.append(i["name"])
    return ext

movies['genres'] = movies['genres'].apply(convert)
movies['genres'] = movies['genres'].apply(lambda x: ", ".join(x))

# creating final dataframe with one feature
movies['combined_df'] = movies.apply(
    lambda row: f"Title: {row['title']}. Overview: {row['overview']}. Genres: {row['genres']}", axis=1
)

# saving final data
movies[['combined_df']].to_csv("data/preprocessed/final_movies_data.csv", index=False, encoding='utf-8')

# laoding the preprocessed data
loader = CSVLoader(file_path="data/preprocessed/final_movies_data.csv", encoding='utf-8')
data = loader.load()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(data)

# Embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docssearch = Chroma.from_documents(texts, embeddings, persist_directory="chroma_store")
docssearch.persist()
retriever = docssearch.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# define llm
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# define prompt
prompt = PromptTemplate(
    template="""You are a movie recommender system that helps users find movie that match their preferences.
Use the following pieces of context to answer the question at the end.
For each question, suggest three movies, with a short description of the plot and the reason why the user might like it.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"]
)

parser = StrOutputParser()

# define chain
chain = prompt | model | parser

# recommend_movie function
def recommend_movie(query):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    result = chain.invoke({"question": query, "context": context})
    return result


# gradio interface
iface = gr.Interface(
    fn=recommend_movie,
    inputs=gr.Textbox(label="Enter your movie preference", lines=5),
    outputs=gr.Textbox(label="Recommended Movies", lines=15),
    title="Movie Recommendation Engine",
    theme="freddyaboulton/darkly@0.0.3"
)

if __name__ == "__main__":
    iface.launch()
