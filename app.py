import cv2
import streamlit as st
import pandas as pd
from deepface import DeepFace

# Function to analyze emotions using DeepFace
def analyze_emotion():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Placeholder for webcam feed

    if "captured_emotions" not in st.session_state:
        st.session_state["captured_emotions"] = []

    stop_button = st.button("Stop Emotion Capture")

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame. Exiting...")
            break

        try:
            # Analyze the frame for emotions
            result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False, silent=True)
            dominant_emotion = result[0]["dominant_emotion"]

            # Overlay dominant emotion on the frame
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Add the emotion to the session state
            st.session_state["captured_emotions"].append(dominant_emotion)

            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

        except Exception as e:
            st.error(f"Error analyzing frame: {e}")

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

# Function to load movies from a file
def load_movies(file_path):
    try:
        movies = pd.read_csv(file_path, header=None, names=["Movie Name"], sep=",")
        movies.dropna(inplace=True)
        movies["Movie Name"] = movies["Movie Name"].astype(str).str.strip().str.rstrip(",")
        return movies["Movie Name"].tolist()
    except Exception as e:
        st.error(f"Error loading movie file: {e}")
        return []

# Combined App: Movies and Emotions Independently
def movie_selector_and_emotion():
    st.title("Movie Selector and Emotion Capture")

    # Section 1: Movie Selector
    st.subheader("Movie Selector")
    uploaded_file = st.file_uploader("Upload your movie file (CSV format with one column for movie names):")
    if uploaded_file:
        movie_list = load_movies(uploaded_file)
        if movie_list:
            # Maintain a session state for selected movies
            st.session_state["selected_movies"] = st.session_state.get("selected_movies", [])

            # Dropdown for movie selection
            movie_selected = st.selectbox("Select a movie to add to your list:", movie_list)

            if st.button("Add Movie"):
                if movie_selected not in st.session_state["selected_movies"]:
                    st.session_state["selected_movies"].append(movie_selected)
                    st.success(f"Added '{movie_selected}' to your list!")

            # Display selected movies
            st.write("Your Selected Movies:")
            for movie in st.session_state["selected_movies"]:
                st.write(f"- {movie}")
        else:
            st.warning("No movies found in the uploaded file.")
    else:
        st.info("Please upload a movie file to proceed.")

    # Section 2: Real-Time Emotion Analysis
    st.subheader("Real-Time Emotion Analysis")
    st.write("Activate your webcam to capture real-time emotions.")
    if st.button("Start Emotion Capture"):
        analyze_emotion()

    # Display captured emotions
    st.subheader("Captured Emotions:")
    if "captured_emotions" in st.session_state and st.session_state["captured_emotions"]:
        for emotion in st.session_state["captured_emotions"]:
            st.write(f"- {emotion}")
    else:
        st.info("No emotions captured yet.")

# Main function
def main():
    st.sidebar.title("Menu")
    activities = ["Movies and Emotions", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Movies and Emotions":
        movie_selector_and_emotion()
    elif choice == "About":
        st.write("This application allows users to select movies and analyze emotions in real-time independently.")
        st.markdown("""
        **Developer:** Shrimanta Satpati  
        **Email:** satpatishrimanta@gmail.com
        """)

if __name__ == "__main__":
    main()

