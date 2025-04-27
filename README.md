# AI Diet Manager

**AI Diet Manager** is a Streamlit application powered by Google Gemini and PHI agents to generate personalized weekly meal plans, calorie recommendations, and food analysis based on user preferences. It supports user authentication, diet plan creation via an interactive questionnaire, food image analysis, and a chat-based assistant for on-the-fly diet modifications.

## Features

- **User Authentication**: Sign up and login with secure password hashing using SHA‑256.
- **Interactive Diet Creation**: Answer a series of guided questions to tailor your meal plan and calorie advice.
- **Weekly Meal Plan**: Automatically generate a week’s worth of meals in strict JSON format via Google Gemini.
- **Calorie Recommendations**: Get daily calorie targets and smart tips.
- **Food Image Analysis**: Upload a dish photo, detect ingredients, compute nutrition per gram, and align it with your diet.
- **Chatbot Assistant**: Converse with a diet assistant to edit plans or ask questions based on your saved diet.
- **Persistent Storage**: All data (users, diets, chat history) is stored in an SQLite database (`data.db`).

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Jnan-py/ai-diet-manager.git
   cd ai-diet-manager
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\\Scripts\\activate   # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Create a `.env` file** in the project root with your API keys:

   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   SERPAPI_API_KEY=your_serpapi_api_key_here
   ```

2. **Ensure your keys are valid**. The app will prompt an error on startup if they are missing.

## Usage

Run the Streamlit app:

```bash
streamlit run main.py
```

- Access the app in your browser at `http://localhost:8501`.
- Use the sidebar to **Sign Up** or **Login**.
- Create or load diet plans, analyze food images, and chat with the assistant!

## Project Structure

```
├── main.py               # Main Streamlit application
├── data.db              # SQLite database (auto-created)
├── .env                 # Environment variables (not tracked)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```
