
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import CharacterTextSplitter
from  dotenv import load_dotenv
from my_summarizer import get_transcript_youtube, summarize_chapters_lexrank
import traceback
import pprint as print

st.markdown("# Home 🎈")
st.sidebar.markdown("# Home 🎈")


video_url = st.text_input("Video URL", value="https://www.youtube.com/watch?v=6CFG84Wo8GM", key=None, type='default')

if st.button('Sumarizar Vídeo'):
    if video_url:
        transcript, _id, chapters, title = get_transcript_youtube(video_url)
        try:
            if chapters:
                 with st.spinner(f"Summarizando o vídeo {title}"):
                    summaries = summarize_chapters_lexrank(chapters)
                    if summaries:
                        st.header(title)
                    else:
                        st.header("Não foi possível sumarizar o vídeo")
                        exit()
                    for summary in summaries:
                        st.subheader(summary['title'])
                        st.write(summary['summary'])
            else:
                st.write("Não foi possível sumarizar o vídeo")    
        except Exception as e:
            st.write(f"Erro ao sumarizar: {e}")
            st.write(traceback.format_exc())  # Print the traceback
    else:
        st.write("Insira uma URL youtube válida")



