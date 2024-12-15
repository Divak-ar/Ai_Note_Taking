import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set page config FIRST
st.set_page_config(page_title="NoteWise", page_icon="📝", layout="wide")

# --- Styling ---
st.markdown(
    """
    <style>
    .big-font {
        font-size: 4rem !important;
        font-weight: 700;
        color: #2980b9;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
    }
    .centered {
        text-align: center;
    }
    .feature-box {
        border: 1px solid #eee;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .feature-title {
        color: #34495e;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Home Page ---
def home_page():
    st.markdown("<p class='centered'><span class='big-font'>NoteWise</span></p>", unsafe_allow_html=True)
    st.markdown("<p class='centered'>Your AI-Powered Study Companion</p>", unsafe_allow_html=True)

    st.write("NoteWise empowers you with intelligent tools to enhance your learning experience:")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):  # Use container for consistent styling
            st.markdown("<h3 class='feature-title'>Assignment Chat</h3>", unsafe_allow_html=True)
            st.markdown("Effortlessly chat with your PDF assignments and get instant answers to your questions.")
            try:
                st.image("images/assignment.png", use_container_width=True)  # Corrected: use_container_width
            except FileNotFoundError:
                st.error("Error: Image 'images/assignment.png' not found.")
                st.info("Place 'assignment.png' in a folder named 'images' in the same directory as your script.")
            except Exception as e:
                st.error(f"An unexpected error occurred loading the assignment image: {e}")

    with col2:
        with st.container(border=True):  # Use container for consistent styling
            st.markdown("<h3 class='feature-title'>YouTube Summarizer</h3>", unsafe_allow_html=True)
            st.markdown("Quickly generate concise summaries of YouTube lectures and tutorials, saving you valuable time.")
            try:
                st.image("images/youtube.png", use_container_width=True)  # Corrected: use_container_width
            except FileNotFoundError:
                st.error("Error: Image 'images/youtube.png' not found.")
                st.info("Place 'youtube.png' in a folder named 'images' in the same directory as your script.")
            except Exception as e:
                st.error(f"An unexpected error occurred loading the youtube image: {e}")

    st.markdown("---")
    st.markdown("<p class='centered'>Start optimizing your study sessions today!</p>", unsafe_allow_html=True)
    

# --- Pdf Chat Page ---
def assignment_chat_page():

    def get_pdf_text(pdf_docs):
        text=""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    #We will divide the text into smaller chunks for vectorization (10000 tokens or words)
    def get_text_chunks(text):
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks=text_splitter.split_text(text)
        return chunks

    def get_vector_store(text_chunks):
        embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
        vector_store.save_local("faiss_index")

    # we store this vector_store in a db but I'll use local env as faiss index folder

    def get_conversational_chain():
        prompt_template="""Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the pdf provided", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        
        Answer: 
        """
        model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

        prompt=PromptTemplate(template=prompt_template, input_variables=["context","question"])
        chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
        return chain


    def user_input(user_question):
        embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Loading out chunks in local env stored in faiss_index folder
        new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
        # Performing similarity search to search for content that is similar to the user question
        docs = new_db.similarity_search(user_question)

        # initializing Convo chain for QnA
        chain=get_conversational_chain()
        response = chain({"input_documents":docs, "question":user_question}, return_only_outputs=True)

        print(response)
        st.write("Reply: ", response["output_text"])


    st.header("Chat With Pdfs 🗣️")
    user_question = st.text_input("Ask a Question from the Pdf")

    if user_question:
        # Taking the user_question as input by calling user_input
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload the PDF Files and Click on the Submit Button",accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing....."):
                # calling text fn to get text data from pdf
                raw_text=get_pdf_text(pdf_docs)
                # Converting that text into chunks (vectorization using Faiss)
                text_chunks=get_text_chunks(raw_text)
                # Storing them chunks in local
                get_vector_store(text_chunks)
                st.success("Done",icon="✅")

    # if __name__ == "__main__":
    #     main()
        

# --- YouTube Summarizer Page ---
def youtube_summarizer_page():
    # Getting the transcript from yt video
    def extract_transcipt_details(youtube_video_url):
        try:
            video_id=youtube_video_url.split("=")[1]
            transcript_text=YouTubeTranscriptApi.get_transcript(video_id)

            transcript = ""
            for i in transcript_text:
                transcript += " " + i["text"]
            return transcript
        except Exception as e:
            raise e


    def generate_gemini_content(transcript_text, subject):
        if subject == "CS":
            prompt = """
                Title: Detailed Computer Science and Software Engineering Notes from YouTube Video Transcript

                As a Computer Science and Software Engineering expert, your task is to provide detailed notes based on the transcript of a YouTube video I'll provide. Assume the role of a student and generate comprehensive notes covering the key concepts discussed in the video.

                Your notes should:

                - Highlight fundamental concepts,algorithms, syntax, and data structure discussed in the video.
                - Explain any relevant concepts,related questions asked in interview,solution to said question, or real-world applications.
                - Clarify any algorithm or method used and provide explanations for their significance along with their time complxity and space complexity.
                - Dicuss their real life applications and better alternative to that code/methods used .
                - Showcase/Generate code if seemed necessary to enhance understanding (preferred mostly if mentioned or used in the video).
                - Do Use diagrams, illustrations, or examples to enhance understanding where necessary.
                - Clarify if any mathematical equations or formulas introduced and provide explanations for their significance.
                - Offer insights into software design patterns, coding best practices, version control systems, and testing methodologies.

                Please provide the YouTube video transcript, and I'll generate the detailed Computer Science and Software Engineering notes accordingly.
            """
        elif subject == "Business Study":
            prompt = """
                Title: Detailed Business Study Notes from YouTube Video Transcript

                As a Business expert, your task is to provide detailed notes based on the transcript of a YouTube video I'll provide. Assume the role of a student and generate comprehensive notes covering the key concepts discussed in the video.You have a deep understanding of various aspects of business operations, management, strategy, finance, marketing, and more.

                Your notes should:

                - Break down terms used and give explain their use in the video.
                - Summarize the key findings and insights gained from the transcript.
                - Include structured data, case studies, market reports, and financial statements for comprehensive learning if ant provided.
                - Draw insights and lessons learned from the video that can be applied in similar contexts or industries.
                - Conclude with actionable takeaways and implications for practitioners, policymakers, or researchers.
                - Provide definitions and explanations of key business concepts such as: Marketing strategies , Financial analysis ,Supply chain      management, Customer relationship management, Market segmentation, Competitive analysis, Business models (e.g., B2B, B2C, etc.),Revenue streams and profitability, Risk management, Strategic planning

            """
        elif subject == "Generate codes":
            prompt = """
                Title: Generate programming codes from YouTube Video Transcript and official documentations

                As a programming expert , your job is to generate and summarized all the code syntax and logic used in the youtube video , to give an accurate answer use your generative skills and official technology/language documentation to provide the code and the syntax used describing and explaning them

                Your notes should:

                - Highlight the code or syntax used and what are used for, also use different colors for codes (like in code editor).
                - Provide a basic idea of the syntax , can refer to the documentations for that technology.
                - Provide alternate methods to achieve the same task , or alternate tools/ technology to implement the same logic.
                - Give it's real life application and industrial use.
                - Try your absolute best to not to provide incorrect information , instead cite the resouces weher we can find them.
                - Give basic functions and methods and other language specific information if described in the video .
                
                In the end section, conclude with language/technology syntax and modules/functions required to achieve the result/project/development-model, not going into in depth implementation of them but the basic use of them (summarize in one line) and things to remember when using them and also resources link to where find them if possible.
                
            """
        elif subject == "Mathematics":
            prompt = """
                Title: Detailed Mathematics Notes from YouTube Video Transcript

                As a mathematics expert, your task is to provide detailed notes based on the transcript of a YouTube video I'll provide. Assume the role of a student and generate comprehensive notes covering the key mathematical concepts discussed in the video.

                Your notes should:

                - Outline mathematical concepts, formulas, and problem-solving techniques covered in the video.
                - Provide step-by-step explanations for solving mathematical problems discussed.
                - Clarify any theoretical foundations or mathematical principles underlying the discussed topics.
                - Include relevant examples or practice problems to reinforce understanding.

                Please provide the YouTube video transcript, and I'll generate the detailed mathematics notes accordingly.
            """
        elif subject == "Data Science and Statistics":
            prompt = """
                Title: Comprehensive Notes on Data Science and Statistics from YouTube Video Transcript

                Subject: Data Science and Statistics

                Prompt:

                As an expert in Data Science and Statistics, your task is to provide comprehensive notes based on the transcript of a YouTube video I'll provide. Assume the role of a student and generate detailed notes covering the key concepts discussed in the video.

                Your notes should:

                Data Science:

                Explain fundamental concepts in data science such as data collection, data cleaning, data analysis, and data visualization.
                Discuss different techniques and algorithms used in data analysis and machine learning, including supervised and unsupervised learning methods.
                Provide insights into real-world applications of data science in various fields like business, healthcare, finance, etc.
                Include discussions on data ethics, privacy concerns, and best practices in handling sensitive data.
                Statistics:

                Outline basic statistical concepts such as measures of central tendency, variability, and probability distributions.
                Explain hypothesis testing, confidence intervals, and regression analysis techniques.
                Clarify the importance of statistical inference and its role in drawing conclusions from data.
                Provide examples or case studies demonstrating the application of statistical methods in solving real-world problems.

                Your notes should aim to offer a clear understanding of both the theoretical foundations and practical applications of data science and statistics discussed in the video. Use clear explanations, examples, and visuals where necessary to enhance comprehension.

                Please provide the YouTube video transcript, and I'll generate the detailed notes on Data Science and Statistics accordingly.
            """
        elif subject == "Case Study":
            prompt = """
                Title: Comprehensive Notes on Case Study from YouTube Video Transcript

                Subject: Case Study

                Objective: The objective of this case study analysis is to examine and explore real-world scenarios across diverse domains, encompassing business, education, research, decision-making, policy development, knowledge sharing, and marketing communication. The aim is to leverage case studies as valuable tools for problem-solving, learning, decision-making, and knowledge dissemination, while also considering their role in informing policy development and supporting marketing efforts.

                Prompt:

                As an expert in Case Study, your task is to provide comprehensive notes based on the transcript of a YouTube video I'll provide. Assume the role of a student and generate detailed notes covering the key concepts discussed in the video.

                Your notes should:

                Case Study:

                Identify and analyze real-world problems or challenges faced by individuals, organizations, or communities across various domains.
                Explore the underlying factors contributing to the identified problems and assess their implications.
                Utilize case study methodology to investigate and analyze complex phenomena, theories, and practical applications across different disciplines.
                Apply qualitative and quantitative research methods to gather and analyze data, including interviews, surveys, observations, and statistical analysis.

                Evaluate the role of case studies as effective teaching and learning tools in academic and professional settings.
                Assess the impact of case studies on student engagement, critical thinking, problem-solving skills, and knowledge acquisition.

                Explore the role of case studies in sharing best practices, lessons learned, and successful strategies across industries, sectors, and communities.
                Assess the effectiveness of different knowledge-sharing platforms and communication channels in disseminating case study findings to diverse audiences.Analyze successful case study campaigns and their impact on brand reputation, customer engagement, and market positioning.

                Conclusion: 

                Conclude the case study analysis with actionable takeaways, insights, and implications for practitioners, policymakers, researchers, educators, business leaders, and other stakeholders across different domains. Emphasize the importance of leveraging case studies as powerful instruments for problem-solving, learning, decision-making, knowledge sharing, and strategic communication in today's dynamic and interconnected world.

                Please provide the YouTube video transcript, and I'll generate the detailed notes on Case Study accordingly.
            """
        else:
            prompt="""You are Youtube video summarizer. You will be taking the transcript text and summarizing the entire video and providing the important summary in points within 500 words or as much as needed to summarize it completely and not miss any important point.Please Provide the summary of the text given here: """


        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt + transcript_text)
        return response.text

    # def generate_gemini_content(transcript_text,prompt):

    #     model=genai.GenerativeModel("gemini-pro")
    #     response=model.generate_content(prompt+transcript_text)
    #     return response.text

    st.title("Youtube Video Summarizer")
    youtube_link = st.text_input("Enter the Video Url")
    subject = st.selectbox("Select Subject:", ["CS", "Business Study", "Generate codes", "Mathematics", "Data Science and Statistics" , "Case Study" , "Youtube"])

    if youtube_link:
        video_id = youtube_link.split('=')[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    if st.button("Get Detailed Summary"):
        transcript_text = extract_transcipt_details(youtube_link)

        if transcript_text:
            summary=generate_gemini_content(transcript_text,subject)
            st.markdown("Summary:")
            st.write(summary)

# --- Main App Logic ---
# PAGES = {
#     "Home": home_page,
#     "Assignment Chat": assignment_chat_page,
#     "YouTube Summarizer": youtube_summarizer_page,
# }

# st.sidebar.title("Navigation")
# selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# page = PAGES[selection]
# page()

# --- App Navigation ---
def main():
    pages = {"Home": home_page, "Assignment Chat": assignment_chat_page, "YouTube Summarizer": youtube_summarizer_page}
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(pages.keys()))
    pages[choice]()

if __name__ == "__main__":
    main()