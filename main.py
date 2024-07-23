from modules.environment.environment_utilities import (
    load_environment_variables,
    verify_environment_variables,
)
from modules.neo4j.credentials import neo4j_credentials
from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser

# Main program
try:
    # Load environment variables using the utility
    env_vars = load_environment_variables()
    

    # Verify the environment variables
    if not verify_environment_variables(env_vars):
        raise ValueError("Some environment variables are missing!")

    llm = OpenAI(
        openai_api_key=env_vars["OPEN_AI_SECRET_KEY"],
        model="gpt-3.5-turbo-instruct",
        temperature=0)
  
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-1.5-pro",
    #     temperature=0,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    #     google_api_key=env_vars["GOOGLE_API_KEY"]
    #     # other params...
    # )

    # Pregunta bàsica contra LLM 
    print("\n\n")
    print("Pregunta bàsica contra LLM sobre Neo4j")
    print("What is Neo4j? \n")
    response = llm.invoke("What is Neo4j?")
    print(response)

    # Ús de template
    template = PromptTemplate(template="""You are a cockney fruit and vegetable seller. Your role is to assist your customer with their fruit and vegetable needs. Respond using cockney rhyming slang. 
                              
    Tell me about the following fruit: {fruit}""", input_variables=["fruit"])

    response = llm.invoke(template.format(fruit="apple"))

    print(" \n")
    print("Pregunta amb template")
    print(response)

    # Output like string
    template = PromptTemplate.from_template("""
                                            You are a cockney fruit and vegetable seller.
                                            Your role is to assist your customer with their fruit and vegetable needs.
                                            Respond using cockney rhyming slang.

                                            Tell me about the following fruit: {fruit}
                                            """)

    llm_chain = template | llm | StrOutputParser()

    response = llm_chain.invoke({"fruit": "apple"})

    print(" \n")
    print("Pregunta amb template amb sortida string")
    print(response)

    # Output like Json
    template = PromptTemplate.from_template("""
                                            You are a cockney fruit and vegetable seller.
                                            Your role is to assist your customer with their fruit and vegetable needs.
                                            Respond using cockney rhyming slang.

                                            Output JSON as {{"description": "your response here"}}

                                            Tell me about the following fruit: {fruit}
                                            """)

    llm_chain = template | llm | SimpleJsonOutputParser()

    response = llm_chain.invoke({"fruit": "apple"})

    print(" \n")
    print("Pregunta amb template amb sortida json")
    print(response)
    print(" \n")

    response = llm_chain.invoke({"fruit": "melocoton"})

    print(" \n")
    print("Pregunta amb template amb sortida json")
    print(response)
    print(" \n")

except Exception as e:
    print(f"An unexpected error occurred: {e}")