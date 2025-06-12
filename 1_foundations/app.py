from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import chromadb
from chromadb.config import Settings
import hashlib
from pathlib import Path

load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Harshit Gulati"
        
        #Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="personal_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load and process documents
        self._load_documents()

    def _chunk_text(self, text, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks for better retrieval."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            if start >= len(text):
                break
        return chunks

    def _process_pdf(self, file_path):
        """Extract text from PDF file."""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def _process_txt(self, file_path):
        """Read text from TXT file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _get_file_hash(self, file_path):
        """Generate hash for file to check if it's already processed."""
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _load_documents(self):
        """Load and process all documents in the 'me' directory."""
        me_dir = Path("me")
        if not me_dir.exists():
            print("'me' directory not found")
            return

        # Get existing document IDs to avoid duplicates
        existing_docs = self.collection.get()
        existing_ids = set(existing_docs['ids']) if existing_docs['ids'] else set()

        for file_path in me_dir.iterdir():
            if file_path.is_file():
                file_hash = self._get_file_hash(file_path)
                doc_id_prefix = f"{file_path.name}_{file_hash}"
                
                # Skip if already processed (check if any chunk from this file exists)
                if any(doc_id.startswith(doc_id_prefix) for doc_id in existing_ids):
                    print(f"Skipping {file_path.name} - already processed")
                    continue

                print(f"Processing {file_path.name}...")
                
                try:
                    # Extract text based on file type
                    if file_path.suffix.lower() == '.pdf':
                        text = self._process_pdf(file_path)
                    elif file_path.suffix.lower() == '.txt':
                        text = self._process_txt(file_path)
                    else:
                        print(f"Unsupported file type: {file_path.suffix}")
                        continue

                    if not text.strip():
                        print(f"No text extracted from {file_path.name}")
                        continue

                    # Split into chunks
                    chunks = self._chunk_text(text)
                    
                    # Add chunks to ChromaDB
                    for i, chunk in enumerate(chunks):
                        doc_id = f"{doc_id_prefix}_chunk_{i}"
                        self.collection.add(
                            documents=[chunk],
                            metadatas=[{
                                "source": file_path.name,
                                "chunk_index": i,
                                "file_type": file_path.suffix.lower()
                            }],
                            ids=[doc_id]
                        )
                    
                    print(f"Added {len(chunks)} chunks from {file_path.name}")
                    
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")

    def _retrieve_relevant_context(self, query, n_results=5):
        """Retrieve relevant document chunks based on the query."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if results['documents'] and results['documents'][0]:
                context_parts = []
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    source = metadata.get('source', 'Unknown')
                    context_parts.append(f"From {source}:\n{doc}")
                
                return "\n\n---\n\n".join(context_parts)
            return ""
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""                

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self, user_message=""):
        # Get relevant context from documents
        relevant_context = self._retrieve_relevant_context(user_message) if user_message else ""
        
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You have access to various documents about {self.name} which contain detailed information about their background, experience, and expertise. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool."

        if relevant_context:
            system_prompt += f"\n\n## Relevant Information:\n{relevant_context}\n\n"
        
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt(message)}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
    