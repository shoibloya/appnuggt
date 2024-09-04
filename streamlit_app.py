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
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import pandas as pd
import time 
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder 
from langchain_core.messages import AIMessage, HumanMessage
import os
import uuid
from datetime import datetime

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = st.secrets['tapiKey']

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets['apiKey']

tools = [TavilySearchResults(max_results=1)]

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

def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("nuggt-nus-firebase-adminsdk-57mm7-b03004fe53.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://nuggt-nus-default-rtdb.firebaseio.com'
        })

def fetch_data_from_firebase():
    init_firebase()
    ref = db.reference('/uploaded_data')  # Adjust this path to your data in Firebase
    data = ref.get()
    if data is None:
        return pd.DataFrame()  # Return an empty DataFrame if no data
    else:
        # Create a DataFrame from a list of dictionaries
        students = [value[0] for key, value in data.items()]
        print(list(data.values())[0])
        return pd.DataFrame(list(data.values())[0])

def check_credentials(username, password):
    correct_username = "emba_nus"
    #correct_password = st.secrets["admin"]
    correct_password = "!ev3CN8z@Pp"
    if username == correct_username and password == correct_password:
        return True
    return False

def login_form():
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if check_credentials(username, password):
                st.session_state['logged_in'] = True  # Update session state
                st.experimental_rerun()  # Rerun the app to update the view
            else:
                st.error("Error: Enter correct username/password")

def active_accounts_view():
    st.subheader("Active Accounts")
    
    # Fetch data from Firebase
    df = fetch_data_from_firebase()
    if len(df) > 0:
        df = df[['Student Name', 'Email']]

    # Display number of accounts and progress
    total_accounts = len(df)
    st.write(f"Number of Active Accounts: {total_accounts} out of 60")
    
    # Progress bar
    progress = total_accounts / 60.0
    st.progress(progress)

    # Display DataFrame
    #if not df.empty:
    #    st.dataframe(df, width=800, height=300)  # DataFrame should now be correctly formatted

def save_data_to_firebase(data):
    ref = db.reference('/uploaded_data')
    # Convert DataFrame to dictionary
    data_dict = data.to_dict(orient='records')
    # Save to Firebase
    ref.push(data_dict)

def start_new_session():
    # Generate a unique session ID
    return str(uuid.uuid4())

def upload_conversation_to_firebase(session_id, chat_data):
    init_firebase()
    ref = db.reference(f'/chat_sessions/{session_id}')
    
    # Adding timestamp for each message
    timestamp = datetime.now().isoformat()
    chat_data['timestamp'] = timestamp
    
    # Push the chat data to the session
    ref.push(chat_data)
    
def main_view():
    st.title("Welcome EMBA!")

    # Displaying user information
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        st.write("Name: EMBA")
        st.write("Email: sirajuddin@nus.edu.sg")
        st.write("Organisation: National University of Singapore")
        active_accounts_view()
        
    with col2:
        st.subheader("Account Details")
        #st.image("demo.png", width=300)
        st.write("Type: Large classroom")
        st.write("Expiry: 5th October 2024")
        st.write("Student Accounts: 60")
        st.subheader("Accessible Tools")
        st.info('Due Diligence 📊')
        st.info('Ideation Tool')
        
    
    st.subheader("Send Student Invitations")
    st.success("Please upload a CSV file containing only two columns, titled 'Student Name' and 'Email'. Once you upload the email list, you will be able to send invitations to students at their email addresses. The invitation will include a link to the app and a PDF with instructions on how to use it.")
   
    # File upload section
    uploaded_file = st.file_uploader("Upload a CSV file with student names and emails", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file into a DataFrame
            data = pd.read_csv(uploaded_file)
            
            # Check if required columns are present
            required_columns = ['Student Name','Email']
            if all(column in data.columns for column in required_columns):
                col1, col2 = st.columns([0.8, 0.2])

                with col1:
                    st.info("CSV file is valid and contains the required columns. Ready to send invitations!")
                
                # Display the DataFrame on the screen
                #st.dataframe(data[required_columns], width=800, height=300)
                
                with col2:
                    # Button to send invitations
                    if st.button('Send Invitation'):
                        st.session_state['button_pressed'] = True
                
                if 'button_pressed' in st.session_state and st.session_state['button_pressed']:
                    save_data_to_firebase(data[required_columns])
                    # Generate and send email to each student
                    st.success("The email list has been successfully uploaded. For security reasons, invitation emails will be sent to all students within the next 24 hours. Upon completion, the number of active students will be updated on your profile. You will receive a notification via email once all invitations have been sent.")
                    #active_accounts_view()
            else:
                st.error("The uploaded CSV file does not contain the required columns 'Student Name' and 'Email'.")
        
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
    
sys_prompt = """
You are a digital assistant designed to guide users through a comprehensive market dynamics analysis to identify opportunities for corporate innovation. Your interaction is structured to ensure a thorough exploration of each market factor one at a time. When you use information from a web source, you always provide the link to the original source

Introduction:
"Welcome to the Market Dynamics Analysis Tool. Let’s identify impactful areas for innovation by analyzing different market factors related to a selected company. Please specify a company we should focus on today together with which area of market dynamics would you like to explore first? Here are your options:
1. Consumer Behavior
2. Economic Conditions
3. Technological Advances
4. Competitive Landscape
5. Regulatory Environment

Step-by-Step Interaction:
1. User selects a market dynamic (e.g., Consumer Behavior).
2. You respond: 'Great choice! What specific questions should we consider to understand changes in consumer behavior for the specified company?'
3. User provides questions.
4. You confirm: 'I will now research the following points: [User’s questions]. Does this sound good?'
5. After user confirmation, you proceed to gather and analyze information.
6. Present your findings and ask if the user wishes to explore another market dynamic.

Conclusion:
After covering all desired aspects, conclude with, 'Based on our analysis, here are some innovative opportunities for the specified company: [summarize opportunities]. What else can I assist you with?'

End each interaction with the phrase, 'This analysis was powered by your dedicated assistant. Let me know how else I can assist you today!'
"""

prompt = ChatPromptTemplate(
    input_variables=['agent_scratchpad', 'input'], # Define the input variables your prompt depends on
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
            input_variables=[], # No input variables for the system message
            template=sys_prompt # Use your new text blob here
            )
        ),
        MessagesPlaceholder(variable_name='chat_history', optional=True), # Include chat history if available
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['input'], # Assuming 'input' is the variable for the human message
                template='{input}' # Use the 'input' variable as the template directly
            )
        ),
        MessagesPlaceholder(variable_name='agent_scratchpad') # Include agent scratchpad messages if available
    ]
)

# Choose the LLM that will drive the agent
# Only certain models support this
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Sidebar navigation
app_mode = st.sidebar.selectbox('Choose the app mode', ['Dashboard', 'Due Diligence', 'Ideation', 'Admin'])

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
        st.info('Ideation')

elif app_mode == "Due Diligence":

    REPORT_PROMPT = "Create the due diligence report. In your report include the following sections\nReport: The information you have gathered from the user in string format. Present it with the relevant subsections.\nFeedback: Feedback to the team based on the information you gathered in string format.\nFinal Decision: Your final decision on whether this is a feasible idea worth the investment in string format."
    st.title("Due Diligence")
    
    # Define a list of companies for selection.
    #company_list = ['Greenfield', 'Google', 'Apple', 'Meta', 'Amazon', 'Microsoft']

    # Capture the selected company each time the user selects from the dropdown.
    #selected_company = st.selectbox("Which company do you want the bot to represent?", company_list)    

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o"

    if "dmessages" not in st.session_state:
        st.session_state.dmessages = [{"role": "system", "content": f"You are a virtual due diligence expert for the specified company. Your task is to gather detailed information about users' new business ideas without revealing the structure or sections of the report. Engage users in a conversational manner, asking one question at a time to ensure clarity. Keep your questions short and to the point. Start by asking them to describe their business opportunity and their rationale on why the company should invest in their business idea. Continue the conversation to gather information on key due diligence findings, assumptions and risks, project overview, market opportunity, strategic alignment, competitive landscape, available resources, technical and business execution feasibility, and the main investment thesis."},
        {"role": "assistant", "content": f"Welcome to the Due Diligence Bot. I'm here to help you clearly articulate and refine your business idea. By asking targeted questions, I'll gather detailed information that can strengthen your proposal. This ensures all aspects of your idea are thoroughly considered, making it more compelling for potential investors. Before we start, please specify the corporate and the innovation."}]
    
    for message in st.session_state.dmessages:
        if message["role"] != "system" and message["content"] != REPORT_PROMPT:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

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

elif app_mode == "Admin":
    init_firebase()
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False  # Initialize session state
    
    if st.session_state['logged_in']:
        main_view()  # Display the main view if logged in
    else:
        st.title("Nuggt Admin Login")
        login_form()  # Display the login form if not logged in

elif app_mode == "Ideation":
    st.title("Ideation chat")

    if "imessages" not in st.session_state:
        introduction = """Welcome to the Market Dynamics Analysis Tool. Let’s identify impactful areas for innovation by analyzing different market factors related to a selected company. Please specify a company we should focus on today together with which area of market dynamics would you like to explore first? Here are your options:

        1. Consumer Behavior
        2. Economic Conditions
        3. Technological Advances
        4. Competitive Landscape
        5. Regulatory Environment
        """
        st.session_state.imessages = [HumanMessage(content="introduce yourself"), AIMessage(content=introduction)]
        
    if "ihistory" not in st.session_state:
        introduction = "Welcome to the Market Dynamics Analysis Tool. Let’s identify impactful areas for innovation by analyzing different market factors related to a selected company. Please specify a company we should focus on today together with which area of market dynamics would you like to explore first? Here are your options:\n1. Consumer Behavior\n2. Economic Conditions\n3. Technological Advances\n4. Competitive Landscape\n5. Regulatory Environment"
        st.session_state.ihistory = [{"role": "assistant", "content": introduction}]
    
    if "session" not in st.session_state:
        st.session_state.session = start_new_session()

    for message in st.session_state.ihistory:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        st.session_state.imessages.append(HumanMessage(content=prompt))
        st.session_state.ihistory.append({"role": "user", "content": prompt})
        upload_conversation_to_firebase(st.session_state.session, {"role": "user", "message": prompt})

        with st.spinner('Thinking...'):
            response = agent_executor.invoke({"input": prompt, "chat_history": st.session_state.imessages})["output"]
       
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.imessages.append(AIMessage(content=response))
        st.session_state.ihistory.append({"role": "assistant", "content": response})
        upload_conversation_to_firebase(st.session_state.session, {"role": "assistant", "message": response})
        
    
