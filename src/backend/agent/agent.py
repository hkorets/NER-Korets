from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.backend.utils.configs import Config
from src.backend.mcp.client import MCPClient

class MyLangChainAgent:
    def __init__(self):
        self.config = Config()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=self.config.openai.API_KEY.get_secret_value()
        )

        self.mcp = MCPClient()

        def _mcp_rag_ask(query: str) -> str:
            res = self.mcp.call_tool("rag_ask", {"query": query})
            return res["content"]["answer"]

        def _mcp_anonymize_text(text: str) -> str:
            res = self.mcp.call_tool("anonymize_text", {"text": text})
            return res["content"]["text"]

        rag_ask_proxy = Tool(
            name="rag_ask",
            description="Answer using RAG (proxied via MCP).",
            func=_mcp_rag_ask,
        )

        anonymize_text_proxy = Tool(
            name="anonymize_text",
            description="Mask entities (proxied via MCP).",
            func=_mcp_anonymize_text,
        )

        self.tools = [anonymize_text_proxy, rag_ask_proxy]

        system_msg = (
            "You are a helpful assistant.\n"
            "- Prefer `anonymize_text` when input contains PII or direct message 'Anonymize'.\n"
            "- Use `rag_ask` for questions about the internal knowledge base about student. Usually starts with 'Find info...'\n"
            "- If neither tool is useful, ANSWER DIRECTLY in your own words.\n"
            "- Keep answers concise unless the user asks for detail."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def run(self, query: str) -> str:
        result = self.executor.invoke({"input": query})
        return result["output"]
