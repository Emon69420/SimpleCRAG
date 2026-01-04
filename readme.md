# CRAG: Contextual Retrieval-Augmented Generation

This project is a Corrective Retrieval-Augmented Generation (RAG) pipeline using Google Gemini and Tavily for web search.

## Requirements

- Python 3.9 or higher
- All dependencies listed in `requirements.txt`

## Setup

1. **Clone the Repository**

   ```
   git clone https://github.com/Emon69420/SimpleCRAG.git
   ```

2. **Install Dependencies**

   ```
   pip install -r requirements.txt
   ```

3. **API Keys**

   You need the following API keys:
   - Google Gemini API Key
   - Tavily API Key

   you can set these keys directly in `app.py` by replacing the empty strings:

   ```python
   api_key = 'your_google_gemini_api_key'
   tavily_api_key = 'your_tavily_api_key'
   ```

4. **Run the Application**

   ```
   python app.py
   ```

   You will be prompted to enter your question in the terminal.

## Project Structure

- `app.py`: Main entry point and workflow
- `classes.py`: Data loading, chunking, vectorization, and retrieval classes
- `corrective.py`: Query rewriting logic
- `evaluator.py`: Document relevance evaluator
- `crag.py`: Example script for running the RAG pipeline

## Notes

- Make sure your API keys are valid and have the necessary permissions.
- The `.env` file should not be committed to version control for security reasons.
- If you encounter issues with missing packages, ensure all dependencies are installed.

## License

This project is for educational purposes.