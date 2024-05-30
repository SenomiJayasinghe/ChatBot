from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import streamlit as st
import os

os.environ['GROQ_API_KEY'] = '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'

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;

    Your turn:

    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    static_responses = {
        "hi": "Hello! How can I assist you today?",
        "hey": "Hey there! What can I do for you?",
        "exit": "Goodbye! Have a great day!",
        "bye": "Goodbye! Have a great day!"
    }

    if user_query.lower() in static_responses:
        return static_responses[user_query.lower()]

    try:
        sql_chain = get_sql_chain(db)

        template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.
        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}"""

        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

        chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=lambda _: db.get_table_info(),
                response=lambda vars: db.run(vars["query"]),
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })

        return response

    except Exception as e:
        # Handle unidentified or error responses
        return "I'm sorry, I couldn't understand your question or process it. Please try asking in a different way. ex: what are the branch details? "

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a Stock Market assistant. Ask me anything from your database."),
    ]

load_dotenv()

st.set_page_config(page_title="Chat with Stock Data", page_icon=":speech_balloon:")

st.title("Chat with Stock Data")

with st.sidebar:
    st.subheader("Settings")
    st.write("Connect to the database and start chatting.")

    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="5432", key="Port")
    st.text_input("User", value="postgres", key="User")
    st.text_input("Password", type="password", value="#cse123", key="Password")
    st.text_input("Database", value="chatbot", key="Database")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))
