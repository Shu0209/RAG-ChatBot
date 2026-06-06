
from flask import Flask, render_template, request, jsonify, session
from src.helper import load_document_from_bytes, text_split, download_hugging_face_embeddings,SUPPORTED_EXTENSIONS
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt
from dotenv import load_dotenv
import os
import uuid

app = Flask(__name__)
app.secret_key = "super_secret_pdf_chatbot_key_2025"  

load_dotenv()


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Please set PINECONE_API_KEY and OPENAI_API_KEY in .env")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "pdfchatbot" 

llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free", 
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.3,
    max_tokens=512,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


@app.route("/")
def index():
    session.clear()  
    return render_template("chat.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    ext = os.path.splitext(file.filename)[1].lower()

    if file.filename == "" or ext not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}"}), 400

    try:
        file_bytes = file.read()
        documents = load_document_from_bytes(file_bytes,file.filename)
        chunks = text_split(documents)

        if len(chunks) == 0:
            return jsonify({"error": "No text could be extracted from the PDF"}), 400

        
        namespace = str(uuid.uuid4())
        session["namespace"] = namespace
        session["filename"] = file.filename

        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name,
            namespace=namespace,
        )

        return jsonify({
            "message": f"PDF '{file.filename}' uploaded and processed successfully!",
            "filename": file.filename
        })

    except Exception as e:
        return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500


@app.route("/clear", methods=["POST"])
def clear_session():
    session.clear()
    return jsonify({"message": "Session cleared"}), 200

@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form.get("msg", "").strip()
    namespace = session.get("namespace")

    if not user_message:
        return jsonify({"response": "Please type a message."})

    if not namespace:
        return jsonify({"response": "Please upload a PDF first before asking questions."})

    try:
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            namespace=namespace
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        result = rag_chain.invoke({"input": user_message})
        answer = result["answer"]

        return jsonify({"response": answer})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "Sorry, something went wrong. Please try again."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)