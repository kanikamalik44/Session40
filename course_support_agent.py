import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_openai_functions_agent, tool, AgentExecutor
from langchain_core.tools.retriever import create_retriever_tool
from langchain_classic.agents import create_tool_calling_agent

raw_documents = [
    Document(page_content="Refund policy: Full refund within 7 days of enrollment if no live class attended. Partial refund within 30 days per program rules.",
             metadata={"source": "refund_policy.md"}),
    Document(page_content="Attendance policy: Minimum 75% attendance is required for certification and placement support.",
             metadata={"source": "attendance_policy.md"}),
    Document(page_content="Batch change policy: Students may request one batch change per cohort. Missing more than three classes without approved leave may delay batch change.",
             metadata={"source": "batch_change_policy.md"}),
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50, chunk_overlap=10
)

split_documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_documents(
    documents= split_documents,
    embedding= embeddings,
    collection_name="helpdesk_policy_docs"
)

retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":2}
)

course_policy_tool = create_retriever_tool(
    retriever = retriever,
    name ="course_policy_tool",
    description="Use this tool to answer questions about official course rules, including refund policy, attendance requirements, and batch change rules."
)

@tool
def get_ticket_status(ticket_id: str) -> str:
    """
    Use this tool when the user asks about the status of a support ticket, refund request, or batch change request by ticket ID.
    """
    FAKE_TICKET_DATABASE = {
    "TKT-2001": "Refund request under review. Expected response in 2 working days.",
    "TKT-2002": "Batch change request approved. New batch starts next Monday."}

    ticket_status = FAKE_TICKET_DATABASE.get(ticket_id)

    if not ticket_status:
        return f"Ticket not found"
    
    return ticket_status

tools = [course_policy_tool,get_ticket_status]

llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0
)

agent = create_tool_caliing_agent