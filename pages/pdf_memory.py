import re
from io import BytesIO
from typing import List

# Modules to Import
import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader


@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


@st.cache_data
def test_embed():
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    # Indexing
    # Save in a Vector DB
    with st.spinner("It's indexing..."):
        index = FAISS.from_documents(pages, embeddings)
    st.success("Embeddings done.", icon="✅")
    return index


st.title("🤖 PPChat personalizado com memória ")
st.markdown(
    """ 
        ####  🗨️ Converse com seus Pdf's wiwicomnversational Buffer Memory`  
        > *powered by [LangChain]('https://langchain.readthedocs.io/en/latest/modules/memory.html#memory') + 
        [OpenAI]('https://platform.openai.com/docs/models/gpt-3-5')*
        ----
        """
)

st.markdown(
    """
    `openai`
    `langchain`
    `tiktoken`
    `pypdf`
    `faiss-cpu`
    
    ---------
    """
)

st.sidebar.markdown(
    """
    ### Passos:
    1. Upload do arquivo(pdf)
    2. Entre com sua chave de API
    3. Pergute!
    
    """
)
if "file" not in st.session_state:
    st.session_state.file = False
    uploaded_file= st.file_uploader("**Upload Your PDF File**", type=["pdf"])
    if uploaded_file:
        name_of_file = uploaded_file.name
        doc = parse_pdf(uploaded_file)
        pages = text_to_docs(doc)
        if pages:
            with st.expander("Show Page Content", expanded=False):
                page_sel = st.number_input(
                    label="Select Page", min_value=1, max_value=len(pages), step=1
                )
                pages[page_sel - 1]
            api = st.text_input(
                "**Enter OpenAI API Key**",
                type="password",
                placeholder="sk-",
                help="https://platform.openai.com/account/api-keys",
            )
            if api:
                # embeddings = OpenAIEmbeddings(openai_api_key=api)
                # # Indexing
                # # Save in a Vector DB
                # with st.spinner("It's indexing..."):
                #     index = FAISS.from_documents(pages, embeddings)
                index = test_embed()
                qa = RetrievalQA.from_chain_type(
                    llm=OpenAI(openai_api_key=api),
                    chain_type="stuff",
                    retriever=index.as_retriever(),
                )

                # Our tool
                tools = [
                    Tool(
                        name="QA System",
                        func=qa.run,
                        description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
                    )
                ]

                prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
                            You have access to a single tool:"""
                suffix = """Begin!"

                {chat_history}
                Question: {input}
                {agent_scratchpad}"""

                prompt = ZeroShotAgent.create_prompt(
                    tools,
                    prefix=prefix,
                    suffix=suffix,
                    input_variables=["input", "chat_history", "agent_scratchpad"],
                )

                if "memory" not in st.session_state:
                    st.session_state.memory = ConversationBufferMemory(
                        memory_key="chat_history"
                    )

                # Chain
                # Zero shot agent
                # Agen Executor
                llm_chain = LLMChain(
                    llm=OpenAI(
                        temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"
                    ),
                    prompt=prompt,
                )
                agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
                agent_chain = AgentExecutor.from_agent_and_tools(
                    agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
                )

                query = st.text_input(
                    "**What's on your mind?**",
                    placeholder="Ask me anything from {}".format(name_of_file),
                )

                if query:
                    with st.spinner(
                        "Generating Answer to your Query : `{}` ".format(query)
                    ):
                        res = agent_chain.run(query)
                        st.info(res, icon="🤖")

                with st.expander("History/Memory"):
                    st.session_state.memory

with st.sidebar:
    st.markdown(
        # '''
        # **Read my blog Post:**
        # [ *How to build a Chatbot with ChatGPT API and a Conversational Memory in Python*](https://medium.com/@avra42/how-to-build-a-chatbot-with-chatgpt-api-and-a-conversational-memory-in-python-8d856cda4542)
        # *Codes are avaialble in this blog post.*
        # '''
        "*Codes with a blog post will be available soon.*"
    )
