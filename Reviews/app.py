# app.py
import os
import requests
import streamlit as st
import openai
from typing import List, Dict

# --- Configuration ---
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# --- Helper Functions ---

def search_app(app_name: str) -> (str, str):
    """Search iTunes API for the app and return the first match's ID and name."""
    url = "https://itunes.apple.com/search"
    params = {"term": app_name, "entity": "software", "limit": 1}
    r = requests.get(url, params=params)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return None, None
    app = results[0]
    return str(app.get("trackId")), app.get("trackName")


def fetch_reviews(app_id: str, country: str = "us", limit: int = 50) -> List[Dict]:
    """Fetch top 'limit' most recent reviews via RSS feed."""
    rss_url = f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostRecent/json"
    resp = requests.get(rss_url)
    resp.raise_for_status()
    entries = resp.json().get("feed", {}).get("entry", [])[1:limit+1]
    reviews = []
    for e in entries:
        reviews.append({
            "title": e["title"]["label"],
            "content": e["content"]["label"],
            "rating": int(e.get("im:rating", {}).get("label", 0))
        })
    return reviews


def analyze_reviews_openai(reviews: List[Dict]) -> (str, str):
    """Use OpenAI to extract themes summary and actionable improvements."""
    # Prepare text block
    text = "\n\n".join([
        f"{i+1}. [{r['rating']}/5] {r['title']} - {r['content']}"
        for i, r in enumerate(reviews)
    ])

    # 1. Summarize themes and sentiment
    theme_messages = [
        {"role": "system", "content": "You are a product growth analyst."},
        {"role": "user", "content": (
            "Summarize the following user reviews:\n\n"
            f"{text}\n\n"
            "Extract 3 key positive themes, 3 key pain points, and an overall sentiment (positive/negative/mixed)."
        )}
    ]
    theme_resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=theme_messages,
        temperature=0.0,
        max_tokens=400
    )
    summary = theme_resp.choices[0].message.content.strip()

    # 2. Propose actionable improvements
    improve_messages = [
        {"role": "system", "content": "You are a product growth analyst."},
        {"role": "user", "content": (
            "Based on these user reviews, list 3 specific, high-impact feature or UX improvements the product team should prioritize:\n\n"
            f"{text}\n\n"
            "Provide each improvement as a short actionable bullet."
        )}
    ]
    imp_resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=improve_messages,
        temperature=0.0,
        max_tokens=200
    )
    improvements = imp_resp.choices[0].message.content.strip()

    return summary, improvements

# --- Streamlit UI ---
st.set_page_config(page_title="App Review Analyzer", layout="centered")
st.title("🎵 iOS App Review Analyzer")

app_name = st.text_input("Enter App Name:")
limit = st.slider("# Reviews to analyze", 10, 50, 50, step=10)

if st.button("Analyze Reviews"):
    with st.spinner("Searching for app..."):
        app_id, found_name = search_app(app_name)
    if not app_id:
        st.error(f"No app found matching '{app_name}'. Please check the name and try again.")
    else:
        st.success(f"Found {found_name} (ID: {app_id})")
        with st.spinner("Fetching reviews..."):
            reviews = fetch_reviews(app_id, limit=limit)
        st.write(f"Fetched {len(reviews)} reviews.")
        with st.spinner("Analyzing with OpenAI..."):
            summary, improvements = analyze_reviews_openai(reviews)
        st.subheader("📝 Themes & Sentiment Summary")
        st.markdown(summary)
        st.subheader("💡 Recommended Improvements")
        st.markdown(improvements)
