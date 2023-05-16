# sourcery skip: use-named-expression
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)
import os
import streamlit as st
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.vectorstores import FAISS
from streamlit_extras.stateful_button import button
from langchain import LLMChain
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()


dirname = f"{os.getcwd()}\docs"
OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENVIROMENT")  # next to api key in console
)

@st.cache_resource
def get_save_faiss_index(chunks : List[Document], collection_name : str, repository_path : str, embeddings ):
    index_file_path = os.path.join(repository_path, f"{collection_name}.faiss")
    if os.path.isfile(index_file_path):
        # Index file exists, load it and return
        return  FAISS.load_local(folder_path='./repository/', index_name= collection_name, embeddings=embeddings)
    db = FAISS.from_documents(chunks, embeddings)
    # Index file doesn't exist, create and return a new index
    db.save_local(collection_name, repository_path)
    # Add some data to the index here
    return db

def search_all_faiss_files(repository_path):
    """
    2. Search all file name .faiss inside "repository" folder return a list
    """
    return [os.path.join(repository_path, f) for f in os.listdir(repository_path) if f.endswith('.faiss')]


def does_index_exist(collection_name, repository_path):
    """
    3. Search if a index(.faiss) exists
    """
    index_file_path = os.path.join(repository_path, f"{collection_name}.faiss")
    return os.path.isfile(index_file_path)

def list_indexes():
    return pinecone.list_indexes()

def index_exists(index_name):
    return index_name in list_indexes()

def create_index(index_name):
    pinecone.create_index(index_name, dimension=1536)
    
@st.cache_data    
def load_documents(dirname):
    loader = DirectoryLoader(dirname)
    return loader.load()

@st.cache_data
def split_documents(_docs,  chunk_size=400):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_documents(_docs)

@st.cache_resource
def vectorize_documents(_docs, index_name):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    ldocs = split_documents(_docs)
    create_index(index_name)
    return Pinecone.from_documents(ldocs, embeddings, index_name=index_name)
    
@st.cache_resource
def load_index(index_name):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)



st.markdown("# Similarity Search ❄️", unsafe_allow_html=True)
st.sidebar.markdown("# Similarity Search ❄️")


# List all collections
collections = search_all_faiss_files("./repository")
if len(collections) > 0:
    selected_collection = st.sidebar.selectbox("Selecione uma coleção", [""] + collections)
    if selected_collection:
        with st.spinner(f"Carregando a coleção: {selected_collection}"):
            index = load_index(selected_collection)
        st.success(f"Coleção {selected_collection} carragada com sucesso!!")


        llmopen = ChatOpenAI(temperature="0", openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
        qa = RetrievalQA.from_chain_type(
                    llm=llmopen,
                    chain_type="stuff",
                    retriever=index.as_retriever(),
                )

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
            # st.session_state.memory = ConversationSummaryBufferMemory(
            #     memory_key="chat_history",
            #     max_token_limit = 1600,
            #     llm=llmopen
            # )

        llm_chain = LLMChain(
            llm=llmopen,
            prompt=prompt,
        )
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
        )

        query = st.text_input(
                    "**O que tem em mente?**",
                    placeholder="Pergunte qualquer coisa contida na {}".format(selected_collection),
                )
        go_chat = st.button("Enviar")
        if query and go_chat:
            with st.spinner(
                "Aguarde enquanto sua IA está respondendo : `{}` ".format(query)
            ):
                res = agent_chain.run(query)
                st.info(res, icon="🤖")

            with st.expander("History/Memory"):
                st.session_state.memory
                
        
with st.sidebar:
    # Initialize session state for create_collection_btn
    if 'create_collection' not in st.session_state:
        st.session_state.create_collection = False

    create_collection_btn = button("Criar coleção", key="btnCollection")

    if create_collection_btn:
        new_collection_name = st.text_input("Nome da nova coleção:")
        if does_index_exist(new_collection_name, "./repository"):
            st.warning("Essa coleção já existe!")
            st.stop()
        #if new_collection_name:
        vectorize_btn = button("Vetorizar documentos", key="btnVectorize")
            # Update the URL query parameters with the new collection name
        if vectorize_btn and new_collection_name != "":
            # Call the vectorize_documents function
            with st.spinner("Lendo os documentos da pasta"):
                docs = load_documents(dirname)
            with st.spinner("Vetorizando os documentos"):
                #vectorize_documents(docs, index_name=new_collection_name)
                get_save_faiss_index(docs,collection_name=new_collection_name,repository_path="./repository")
                # Reset the state
            st.success("Documento vetorizado com sucesso, sua sessão está pronta!")
            # st.session_state.create_collection = False
            # st.session_state.collection = False
            st.experimental_rerun()
        else:
            st.sidebar.warning("Entre com o nome da coleção!.")



with st.sidebar:
    st.markdown(
        '''
        1. Coloque todos os documentos que quer estudar na pasta DOCS
        2. Crie ou selecione uma coleçào de documentos
        3. *Para criar uma nova:*
            - clique no botão, entre com o nome
            - Clique no botão "Vetorizar!
        '''
    )



if "vetorize" in st.session_state:
    st.text(st.session_state.vetorize)
    
    