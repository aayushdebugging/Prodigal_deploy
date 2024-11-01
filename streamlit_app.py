import streamlit as st
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util

# Set your Gemini API key here
api_key = "AIzaSyB_bDHH-vwENqIg-Kw4vgfF1FhLwge9jN8" 
# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load tool data from a CSV file
@st.cache_data
def load_tool_data(csv_file):
    """Loads tool data from a CSV file and generates embeddings."""
    df = pd.read_csv(csv_file)
    tool_descriptions = df['description'].tolist()
    tool_embeddings = model.encode(tool_descriptions, convert_to_tensor=True)
    return df, tool_embeddings

# Function to break down a user query into distinct subtasks
def break_down_query(query):
    """Breaks down a user query into distinct, actionable subtasks aimed at achieving the task."""
    prompt = (
        f"I am working on the following marketing task: '{query}'. "
        f"Please generate a list of 5 to 7 unique, clear, and actionable subtasks to effectively complete this task. "
        f"Each subtask should be a standalone action that directly contributes to achieving the goal, free from vague steps or sub-level tasks like 'a', 'b', 'c'. "
        f"Focus on specific actions that can be independently accomplished without requiring further breakdown. "
        f"Present each subtask as a numbered list with one sentence per step for clarity."
    )


    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        response_json = response.json()
        subtasks = response_json.get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text", "").split("\n")
        return [subtask.strip() for subtask in subtasks if subtask.strip() and not subtask.strip().startswith(('a)', 'b)', 'c)'))]
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return []

# Function to search tools with embeddings based on a query and apply weighted scoring
def search_tools_with_embeddings(query, df, tool_embeddings):
    """Search for tools using embeddings based on a query with weighted scoring."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, tool_embeddings)

    results = []
    for idx, score in enumerate(similarities[0]):
        results.append((df['tool_name'][idx], df['website_link'][idx], df['description'][idx], float(score), df['ratings'][idx], df['admin_rating'][idx], df['user_reviews'][idx]))

    top_10_by_similarity = sorted(results, key=lambda x: x[3], reverse=True)[:10]

    weighted_results = []
    for tool in top_10_by_similarity:
        embedding_score = tool[3]
        user_rating_score = tool[4] * 0.3
        admin_rating_score = tool[5] * 0.5
        review_embedding = model.encode(tool[6], convert_to_tensor=True)
        review_similarity = util.pytorch_cos_sim(query_embedding, review_embedding).item() * 0.2
        total_score = embedding_score + user_rating_score + admin_rating_score + review_similarity
        weighted_results.append((tool[0], tool[1], tool[2], total_score, tool[4]))

    top_5_by_weighted_score = sorted(weighted_results, key=lambda x: x[3], reverse=True)[:5]
    return top_5_by_weighted_score

# Load tool data
csv_file = 'Voyex_Tools_Data_with_Updated_User_Reviews.csv'  # Update this path
df, tool_embeddings = load_tool_data(csv_file)

# Streamlit app UI
st.title("Marketing Task Subtasks and Tool Recommendation")

# Input for user query
user_query = st.text_input("Enter your marketing task or goal:")

if user_query:
    st.write("### Generated Subtasks")
    subtasks = break_down_query(user_query)

    if subtasks:
        for subtask in subtasks:
            st.write(f"- **{subtask}**")
            st.write("#### Recommended Tools:")
            tools = search_tools_with_embeddings(subtask, df, tool_embeddings)
            for tool in tools:
                st.write(f"  - **{tool[0]}** ([Link]({tool[1]})): {tool[2]}")
                st.write(f"    - **Score**: {tool[3]:.4f}, **User Rating**: {tool[4]}")
    else:
        st.warning("No subtasks generated. Try rephrasing your query.")
