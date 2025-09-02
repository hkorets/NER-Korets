from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from src.backend.utils.configs import Config

class RAG:

    def __init__(self, path= "data/docs/context.md"):
        self.config = Config()
        self.model = ChatOpenAI(
            api_key=self.config.openai.API_KEY.get_secret_value(),
            model="gpt-4o-mini",
            temperature=0
        )
        self.path = path

    def retriever(self):
        loader = TextLoader(self.path, encoding="utf-8")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(api_key=self.config.openai.API_KEY.get_secret_value())
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    def answer_question(self, question: str):
        retriever = self.retriever()
        prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided context to answer. "
                   "If unsure, say you don't know.\n\nContext:\n{context}"),
        ("human", "{question}")
        ])

        llm = self.model

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        return chain.invoke(question).content