import os
import time
import logging
import hashlib
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableAssign, RunnableLambda
import chromadb
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]


class AnswerResponse(BaseModel):
    answers: List[str]


class OptimizedRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.client = None
        self.splitter = None
        self.prompt = None
        self.initialized = False
        self.doc_cache: Dict[str, Any] = {}
        self.max_cache = 5

    def _hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def initialize(self):
        if self.initialized:
            return

        self.chat_model = ChatTogether(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            temperature=0,
            max_tokens=3500,
        )

        self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")

        path = os.path.join(os.getcwd(), "vectorstore")
        os.makedirs(path, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=path,
            settings=chromadb.Settings(anonymized_telemetry=False)
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.prompt = ChatPromptTemplate([
            ("system",
             "Answer concisely. Questions separated by ' | '. "
             "Return answers separated by ' | ' in same order."),
            ("human", "Questions: {query}\nContext: {context}"),
        ])

        self.initialized = True

    def _load_doc(self, url: str):
        if url in self.doc_cache:
            return self.doc_cache[url]

        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)

        if len(self.doc_cache) >= self.max_cache:
            self.doc_cache.pop(next(iter(self.doc_cache)))

        self.doc_cache[url] = (docs, chunks)
        return docs, chunks

    def _vectorstore(self, url: str, chunks):
        name = f"doc_{self._hash(url)}"

        try:
            col = self.client.get_collection(name)
            if col.count() > 0:
                return Chroma(
                    client=self.client,
                    collection_name=name,
                    embedding_function=self.embeddings,
                )
        except Exception:
            pass

        vs = Chroma(
            client=self.client,
            collection_name=name,
            embedding_function=self.embeddings,
        )

        def add(batch):
            vs.add_documents(batch)

        with ThreadPoolExecutor(max_workers=3) as ex:
            for i in range(0, len(chunks), 50):
                ex.submit(add, chunks[i:i + 50])

        return vs

    def _chain(self, retriever):
        def retrieve(state):
            res = retriever.invoke(state["query"], k=6)
            ctx = " ".join(d.page_content for d in res)
            return ctx[:3500]

        return (
            RunnableAssign({"context": RunnableLambda(retrieve)})
            | self.prompt
            | self.chat_model
            | StrOutputParser()
        )

    def run(self, url: str, questions: List[str]) -> List[str]:
        docs, chunks = self._load_doc(url)
        vs = self._vectorstore(url, chunks)
        retriever = vs.as_retriever(search_kwargs={"k": 6})
        chain = self._chain(retriever)

        query = " | ".join(questions)
        result = chain.invoke({"query": query})
        answers = [a.strip() for a in result.split("|")]

        if len(answers) != len(questions):
            answers += ["Unable to answer"] * (len(questions) - len(answers))

        return answers[:len(questions)]


rag = OptimizedRAGEngine()
app = FastAPI(title="Optimized RAG API")


@app.on_event("startup")
async def startup():
    rag.initialize()


@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    if not req.documents.startswith(("http://", "https://")):
        raise HTTPException(400, "Invalid URL")

    if not req.questions:
        raise HTTPException(400, "No questions")

    loop = asyncio.get_event_loop()
    answers = await loop.run_in_executor(None, rag.run, req.documents, req.questions)
    return {"answers": answers}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "initialized": rag.initialized,
        "cached_docs": len(rag.doc_cache),
        "time": datetime.now().isoformat(),
    }


@app.post("/clear-cache")
async def clear_cache():
    rag.doc_cache.clear()
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
