from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent, Tool
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from src.config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, DEFAULT_MODEL
from src.vector_db import get_vector_db
from src.tools import get_web_search_tool

def get_agent():
    # LLM Initialization (pointing to LM Studio)
    llm = ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        model=DEFAULT_MODEL,
        temperature=0.7
    )

    # RAG Tool with Error Handling
    vector_db = get_vector_db()
    
    def rag_search(query: str):
        if vector_db is None:
            return "Knowledge base is currently unavailable. Please check if the dataset is correctly loaded."
        try:
            docs = vector_db.similarity_search(query, k=3)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"

    rag_tool = Tool(
        name="HouseKnowledgeBase",
        func=rag_search,
        description="Useful for when you need to answer questions about real estate and houses in Nepal from the local dataset."
    )

    # Web Search Tool
    search_tool = get_web_search_tool()
    
    tools = [rag_tool, search_tool]

    # Re-Act Prompt Template (Strict)
    template = """You are a professional Nepal Real Estate Agent. Answer the following questions as best you can.
You have access to the following tools:

{tools}

STRICT FORMATTING RULES:
You MUST follow the Thought/Action/Action Input/Observation cycle precisely.
Each keyword MUST start on a NEW LINE.
Do NOT include any text before 'Thought:'.

To use a tool, use THIS format:
Thought: your reasoning about why you need the tool
Action: the action to take, should be one of [{tool_names}]
Action Input: the EXACT input string for the action (no quotes)
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)

When you have the final answer, use THIS format:
Thought: I have found the answer or do not need more tools.
Final Answer: the final answer to the original question

IMPORTANT: If you can answer based on your current knowledge, just provide Thought and Final Answer.

Begin!

Previous conversation history:
{chat_history}

Question: {input}
{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    # Create the Re-Act Agent using the langchain_classic version
    agent = create_react_agent(llm, tools, prompt)

    # Memory from langchain_classic
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Custom Error Handler for Parsing Failures
    def _handle_error(error) -> str:
        return (
            "ERROR: Formatting Failure. You MUST use 'Thought:', 'Action:', and 'Action Input:' on separate lines.\n"
            "If you are attempting to give a final response, you MUST use 'Final Answer:' on a new line.\n"
            "Please try again using the correct format."
        )
    # Agent Executor from langchain_classic
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=_handle_error,
        return_intermediate_steps=True
    )

    return agent_executor
