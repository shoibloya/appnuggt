import streamlit as st
from openai import OpenAI as OA
from PIL import Image
import io
import base64
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
import tempfile

# Initialize the OpenAI client with your API keys
client = OA(api_key=st.secrets['apiKey'])  # Replace with your actual API key
st.set_page_config(page_title="Nuggt Dashboard")

# Function to simulate bot response, considering both images and text
def get_bot_response():
    # Convert messages for the API call, handling text and images differently
    messages_for_api = []
    for m in st.session_state.messages:
        if m['role'] == 'user' and 'is_photo' in m and m['is_photo']:
            # For images, convert to base64 and create the proper structure
            image_base64 = image_to_base64(m['content'])
            image_url = "data:image/jpeg;base64," + image_base64
            messages_for_api.append({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_url}}]
            })
        else:
            # For text messages, keep the structure simple
            messages_for_api.append({
                "role": m['role'],
                "content": [{"type": "text", "text": m['content']}]
            })

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",  # Use the correct model that supports images
        messages=messages_for_api,
        max_tokens=4000,
    )

    # Append the bot's response to the session state messages
    if response.choices:
        st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        with st.chat_message(st.session_state['messages'][-1]['role']):
            st.markdown(st.session_state['messages'][-1]['content'])

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            loader = PyMuPDFLoader(tmp.name)
            pages = loader.load_and_split()
            db = FAISS.from_documents(pages, OpenAIEmbeddings(openai_api_key=openai_api_key))
            # Create retriever interface
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
            # Create QA chain
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
            print("Answer generated")
            return qa.run(query_text)

def image_to_base64(image):
    # Convert RGBA to RGB
    if image.mode in ("RGBA", "LA"):
        background = Image.new(image.mode[:-1], image.size, (255, 255, 255))
        background.paste(image, image.split()[-1])
        image = background

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_report(client, conversation):
    # This function simulates generating a report based on the conversation.
    # Replace this with your actual implementation using the OpenAI API or any other model.
    # Example:
    response = client.completions.create(
        model="text-davinci-003",  # Use an appropriate model
        prompt=conversation,
        temperature=0.7,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text 


# Sidebar navigation
app_mode = st.sidebar.selectbox('Choose the app mode', ['Dashboard', 'Multimodal', 'Knowledge RAG', 'Due Diligence'])

import streamlit as st

# Assume this section is part of a larger application where app_mode is defined
if app_mode == "Dashboard":
    st.title("Welcome!")

    # Displaying user information
    col1, col2 = st.columns(2)

    with col1:
        st.header("Account Details")
        st.write("Institution: National University of Singapore")
        st.write("Department: EMBA")

    with col2:
        st.header("Accessible Tools")
        st.info('Due Diligence 📊')
        st.info('Knowledge Graph RAG 🔍')
        st.info('Multimodal Playground 🎮')
    

elif app_mode == "Multimodal Playground":
    # Main app layout
    st.title("GPT4 Vision")

    # Inject custom CSS with st.markdown to make a custom container sticky
    st.markdown("""
        <style>
        .sticky-container {
            position: -webkit-sticky; /* For Safari */
            position: sticky;
            top: 0;
            z-index: 1000;
            background-color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    # Create a sticky container for the file uploader
    with st.container():
        st.markdown('<div class="sticky-container">', unsafe_allow_html=True)
        user_input = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        st.markdown('</div>', unsafe_allow_html=True)

    # Initialize session state for messages if not already set
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Chat interface
    for message in st.session_state['messages']:
        if 'is_photo' in message and message['is_photo']:
            pass
            #with st.chat_message(message['role']):
            #    st.image(message['content'])  # Adjust width as needed
        else:
            with st.chat_message(message['role']):
                st.write(message['content'])

    # Check if a file is uploaded
    if user_input is not None:
        # Convert the uploaded file to a PIL Image
        pil_image = Image.open(user_input)
        # Add photo to messages as an item
        st.session_state['messages'].append({'role': 'user', 'content': pil_image, 'is_photo': True})
        
    # Text input for questions
    text_input = st.chat_input("Type your message...", key="chat_input")
    if text_input:
        # Add user message to chat
        st.session_state['messages'].append({'role': 'user', 'content': text_input})
        with st.chat_message(st.session_state['messages'][-1]['role']):
            st.markdown(st.session_state['messages'][-1]['content'])  # Adjust width as needed
        # Trigger bot response
        get_bot_response()

elif app_mode == "Knowledge Graph Retrieval":
    st.title('PDF QnA')

    # File upload
    uploaded_file = st.file_uploader('Upload an article', type='pdf')
    # Query text
    query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

    # Form input and query
    result = []
    with st.form('myform', clear_on_submit=True):
        openai_api_key = st.secrets['apiKey']
        submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
        if submitted and openai_api_key.startswith('sk-'):
            with st.spinner('Calculating...'):
                response = generate_response(uploaded_file, openai_api_key, query_text)
                result.append(response)
                del openai_api_key

    if len(result):
        st.info(response)

elif app_mode == "Due Diligence":

    REPORT_PROMPT = "Create the due diligence report. In your report include the following sections\nReport: The information you have gathered from the user in string format. Present it with the relevant subsections.\nFeedback: Feedback to the team based on the information you gathered in string format.\nFinal Decision: Your final decision on whether this is a feasible idea worth the investment in string format."
    st.title("Due Diligence")
    
    # Define a list of companies for selection.
    company_list = ['Greenfield', 'Google', 'Apple', 'Meta', 'Amazon', 'Microsoft']

    # Capture the selected company each time the user selects from the dropdown.
    selected_company = st.selectbox("Which company do you want the bot to represent?", company_list)    

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4-0125-preview"

    if "dmessages" not in st.session_state:
        st.session_state.dmessages = [{"role": "system", "content": f"You are a virtual due diligence expert for Greenfield innovation projects. Your task is to gather detailed information about users' new business ideas without revealing the structure or sections of the report. Engage users in a conversational manner, asking one question at a time to ensure clarity. Keep your questions short and to the point. Start by asking them to describe their business opportunity and their rationale on why the due diligence team should invest in their business idea. Continue the conversation to gather information on key due diligence findings, assumptions and risks, project overview, market opportunity, strategic alignment, competitive landscape, available resources, technical and business execution feasibility, and the main investment thesis."}]
    
    for message in st.session_state.dmessages:
        if message["role"] != "system" and message["content"] != REPORT_PROMPT:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        elif message["role"] == "system" and message["content"] != REPORT_PROMPT:
            if selected_company != "Greenfield":
                message["content"] = f"You are a virtual due diligence expert for {selected_company}. Your task is to gather detailed information about users' new business ideas without revealing the structure or sections of the report. Engage users in a conversational manner, asking one question at a time to ensure clarity. Keep your questions short and to the point. Start by asking them to describe their business opportunity and their rationale on why the company should invest in their business idea. Continue the conversation to gather information on key due diligence findings, assumptions and risks, project overview, market opportunity, strategic alignment, competitive landscape, available resources, technical and business execution feasibility, and the main investment thesis."
            else:
                message["content"] = f"You are a virtual due diligence expert for Greenfield innovation projects. Your task is to gather detailed information about users' new business ideas without revealing the structure or sections of the report. Engage users in a conversational manner, asking one question at a time to ensure clarity. Keep your questions short and to the point. Start by asking them to describe their business opportunity and their rationale on why the due diligence team should invest in their business idea. Continue the conversation to gather information on key due diligence findings, assumptions and risks, project overview, market opportunity, strategic alignment, competitive landscape, available resources, technical and business execution feasibility, and the main investment thesis."

    
    if len(st.session_state.dmessages) >= 19:
        st.info("You can now generate the due diligence report.")
        if st.button("Generate Report"):
            st.session_state.dmessages.append({"role": "user", "content": REPORT_PROMPT})
            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.dmessages
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
            st.session_state.dmessages.append({"role": "assistant", "content": response})
    
    elif len(st.session_state.dmessages) != 1:
        st.info(f"I will ask {int((19-len(st.session_state.dmessages))/2)} more questions before generating the report.")

    if prompt := st.chat_input("Write your message..."):
        st.session_state.dmessages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.dmessages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.dmessages.append({"role": "assistant", "content": response})
