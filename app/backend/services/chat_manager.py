import os
from backend.models.schemas import LLMResponse
from backend.services.embedding_manager import EmbeddingManager
from backend.services.retrieval_manager import RetrievalManager
from langchain_google_genai import GoogleGenerativeAI
from backend.services.redis_manager import RedisManager
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda

"""Handle logic for chat interactions between LLM and user"""

class ChatManager: 

    @staticmethod
    def prompt_llm(session_id: str, prompt: str, timestamp: str) -> LLMResponse: 
        """
        1. Store user's message in Redis
        2. Retrieve relevant documents from vector store 
        4. Generate LLM response and store in Redis
        """

        # ------------------Store user's message in Redis---------------
        redis_manager = RedisManager()
        redis_manager.add_message(
            session_id=session_id, 
            role="user", 
            content=prompt, 
            timestamp=timestamp
        )

        # ------------------Retrieve relevant documents from vector store---------------
        retriever_manager = RetrievalManager()
        retrieved_docs = retriever_manager.retrieve_documents(prompt)

        # Context to inject into prompt 
        context = "\n\n".join([doc["page_content"] for doc in retrieved_docs["documents"]])

        # ------------------Generate LLM response and store in Redis---------------

        # Instantiate LLM
        llm = GoogleGenerativeAI(model="gemini-1.5-flash")

        # Define prompt template 
        rag_template = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant tasked with using the provided context to answer the question **if it's relevant**. 
                If the question is casual, conversational, or unrelated, respond naturally without relying on the context."""),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

        # Helper function 
        def llm_response_to_redis(response: str) -> LLMResponse: 
            message, timestamp = redis_manager.add_message(
                session_id=session_id, 
                role="assistant", 
                content=response
            )

            return LLMResponse(llmResponse=message, timestamp=timestamp)

        # Define runnables
        format_rag_prompt = RunnableLambda(lambda inputs: rag_template.format_prompt(**inputs))
        invoke_llm = RunnableLambda(lambda prompt_val: llm.invoke(prompt_val.to_messages()).content)
        store_llm_response = RunnableLambda(llm_response_to_redis)
        
        # Assemble chain
        chain = format_rag_prompt | invoke_llm | store_llm_response 
        llm_response = chain.invoke({
            "context": context, 
            "question": prompt
        })

        return llm_response

    @staticmethod
    def upload_file(file): 
        """
        1. Save uploaded file in temporary local directory
        2. Create and store embeddings for file 
        """

        # ------------------Save uploaded file---------------

        # Get upload directory 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
        upload_dir = os.path.join(root_dir, "uploaded_files")

        # Create file path
        file_path = os.path.join(upload_dir, file.filename)

        # Save file in upload directory
        with open(file_path, "wb") as f: 
            f.write(file.file.read())

        # ------------------Create and store embeddings for file ---------------

        print(f"CREATING EMBEDDINGS FOR FILE AT {file_path}....")

        embedding_manager = EmbeddingManager()
        response = embedding_manager.process_uploaded_file(file_path)

        return response