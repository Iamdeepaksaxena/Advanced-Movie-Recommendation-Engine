# 🎥 Advanced Movie Recommendation Engine

> Advanced AI Movie Recommendation Engine built with **LangChain**, **OpenAI**, **HuggingFace Embeddings**, and **Gradio** — delivering context-aware, intelligent, and explainable movie suggestions using Retrieval-Augmented Generation (RAG).

---

## 🎬 Demo
🎞️ Watch the demo below:
<video src="https://github.com/Iamdeepaksaxena/Advanced-Movie-Recommendation-Engine/raw/main/assets/demo.mp4" width="720" controls></video>

> 💡 If GitHub doesn’t render the video inline, [click here to download or view it](https://github.com/Iamdeepaksaxena/Advanced-Movie-Recommendation-Engine/raw/main/assets/demo.mp4).

---

## 🚀 Features
- 🎬 Intelligent movie suggestions based on natural language input  
- 🧠 Uses **LangChain** and **OpenAI GPT-4o-mini** for reasoning  
- 🔍 **HuggingFace Embeddings** + **Chroma** vector store for semantic search  
- 💬 Simple, interactive **Gradio interface**  
- 📂 Works directly from your `movies.csv` dataset  

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```
git clone https://github.com/Iamdeepaksaxena/Advanced-Movie-Recommendation-Engine.git
cd Advanced-Movie-Recommendation-Engine
2️⃣ Create a Virtual Environment (Recommended)
python -m venv venv
.\venv\Scripts\activate   # For Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Environment Variables
Create a .env file in your project root and add:
OPENAI_API_KEY=your_openai_api_key_here

🧩 Run the App
python recommendation.py
