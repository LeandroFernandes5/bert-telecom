import streamlit as st
import requests
import json

# Set up the Streamlit UI
st.title("How was your International Roaming experience with Verizon service?")

# Text input box
user_input = st.text_area("")

# Button to trigger classification
if st.button("Classify"):
    if user_input.strip() == "":
        st.error("Please enter some text to classify.")
    else:
        # Prepare the data for the API request
        data = {"answer": user_input}
        headers = {"Content-Type": "application/json"}
        data['unique_id'] = 123

        # Send the JSON request to the API
        response = requests.post("http://localhost:8001/classify", data=json.dumps(data), headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            sentiment = result.get("sentiment")
            unique_id = result.get("unique_id")

            # Determine the sentiment label
            if sentiment == 0:
                sentiment_label = "Negative"
                color = "red"
            elif sentiment == 1:
                sentiment_label = "Neutral"
                color = "yellow"
            elif sentiment == 2:
                sentiment_label = "Positive"
                color = "green"
            else:
                sentiment_label = "Unknown"
                color = "gray"

            # Display the result
            st.markdown(f"<p style='color:{color};'>Sentiment: {sentiment_label}</p>", unsafe_allow_html=True)
           
        else:
            st.error(f"Error: Received status code {response.status_code} from the API.")