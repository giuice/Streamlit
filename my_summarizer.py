
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import CharacterTextSplitter
from  dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate, LLMChain


from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import spacy
from langchain.text_splitter import SpacyTextSplitter
from sentence_transformers import SentenceTransformer, util
import numpy as np
from LexRank import degree_centrality_scores

from transformers import T5Tokenizer, T5ForConditionalGeneration
# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

load_dotenv()
#bertModel = SBertSummarizer('paraphrase-MiniLM-L6-v2')
# Load the tokenizer and model
# model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
# tok = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# # Initialize the Summarizer with the SBert model and tokenizer
# bertModel = Summarizer(custom_model=model, custom_tokenizer=tok)





llm = OpenAI(temperature=0,model_name="text-babbage-001")

def get_video_title(id):
    link_url = f"https://yt.lemnoslife.com/noKey/videos?part=snippet&id={id}"
    video_snippets = requests.get(link_url).json()
    return video_snippets['items'][0]['snippet']['title']

def get_video_chapters(id):
    link = f"https://yt.lemnoslife.com/videos?part=chapters&id={id}" 
    video = requests.get(link).json()
    return video["items"][0]["chapters"]["chapters"]

def get_transcript_youtube(video_url):
    _id = video_url.split("=")[1].split("&")[0]
    filename = f"files/transcript_{_id}.json"
    transcript = YouTubeTranscriptApi.get_transcript(_id)
    #print(transcript)
    #get chapters
    chapters = get_video_chapters(_id)
    for item in chapters:
        del item["thumbnails"]
    
    for i, chapter in enumerate(chapters):
        chapter['content'] = ''
        chapter_end = chapters[i+1]['time'] if i+1 < len(chapters) else float('inf')
        for transcription in transcript:
            if chapter['time'] <= transcription['start'] < chapter_end:
                chapter['content'] += transcription['text'] + ' '
            elif transcription['start'] >= chapter_end:
                break
    return transcript, _id, chapters, get_video_title(_id)

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

def summarize_chapters(chapters):
    summaries = []
    for chapter in chapters:
        try:
            chapter['content'] = preprocess_text(chapter['content'])
            text_splitter_char = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, chunk_overlap=20)
            texts = text_splitter_char.split_text(chapter['content'])
            docs = [Document(page_content=t) for t in texts]
            chain = load_summarize_chain(llm, chain_type="refine",verbose=True)
        
            summary = chain.run(docs)
        except Exception:
            summary = f"No summary was generated.{Exception}"
        summaries.append({"title":chapter['title'], "summary": summary})
    return summaries


def summarize_chapters_bert(chapters):
    summaries = []
    from summarizer import Summarizer
    model = Summarizer()
    for chapter in chapters:
        # try:
        #chapter['content'] = preprocess_text(chapter['content'])
        text_splitter_char = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, chunk_overlap=20)
        texts = text_splitter_char.split_text(chapter['content'])
        #summary = bertModel('\n'.join(texts), min_length=100, num_sentences=5)
        summary = ''
        for text in texts:
            summary = model(text, min_length=100, num_sentences=5) + "\n" + summary
            
        #summary = model(chapter['content'], min_length=40, num_sentences=3)
        #docs = [Document(page_content=t) for t in texts]
        #summary = [bertModel(t, num_sentences=5) for t in texts]
        # Join the sentences into a single string for each summary
        #summary_text = '\n'.join(summary)
        #summary_text = bertModel(summary_text, num_sentences=5)
        #summary = ''.join(summary)
        summaries.append({"title": chapter['title'], "summary": summary})
    return summaries


def summarize_t5(model, tokenizer,  text, max_length=50):
    input_ids = tokenizer.encode(
        f"summarize: {text}",
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    summary_ids = model.generate(input_ids, num_return_sequences=1, max_length=max_length, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_chapters_t5(chapters):
    '''
    This example uses the t5-small model, but you can try other T5 variants such as 
    t5-base, t5-large, t5-3B, or t5-11B for better performance at the cost of increased computational requirements.
    '''
    model_name = 't5-large'

    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=800)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    summaries = []

    for chapter in chapters:
        text_splitter_char = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, chunk_overlap=20)
        texts = text_splitter_char.split_text(chapter['content'])

        # Combine texts into a single string
        combined_text = '\n'.join(texts)
        summary = ''
        for text in texts:
            summary = summarize_t5(model, tokenizer, text, max_length=300)   + "\n" + summary 
        # Generate the summary
    
        summaries.append({"title": chapter['title'], "summary": summary})

    return summaries

def summarize_chapters_lexrank(chapters):
    summaries = []
    model = SentenceTransformer('all-MiniLM-L6-v2')
    text_splitter = SpacyTextSplitter(chunk_size=400, chunk_overlap=0)
    text_splitter_char = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, chunk_overlap=20)
    for chapter in chapters:
        #sentences = text_splitter.split_text(chapter['content'])
        sentences = text_splitter_char.split_text(chapter['content'])
        #Compute the sentence embeddings
        embeddings = model.encode(sentences, convert_to_tensor=True)

        #Compute the pair-wise cosine similarities
        cos_scores = util.cos_sim(embeddings, embeddings).numpy()

        #Compute the centrality for each sentence
        centrality_scores = degree_centrality_scores(cos_scores, threshold=None)

        #We argsort so that the first element is the sentence with the highest score
        most_central_sentence_indices = np.argsort(-centrality_scores)
        temp_summary = [
            sentences[idx].strip() for idx in most_central_sentence_indices[:4]
        ]
        summaries.append({"title":chapter['title'], "summary": " ".join(temp_summary)})
        
    return summaries