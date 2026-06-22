from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()
import os

core_version=version("langchain-core")
lg_version=version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')

print(f"Langchain_core version: {core_version} ")
print(f"LangGraph version: {lg_version}") 


def main():
    llm = ChatOpenAI(model="gpt-4o-mini")
    response=llm.invoke("Say 'setup completed'")
    print(response)

   



if __name__ == "__main__":
    main()
