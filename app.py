import streamlit as st
import pandas as pd

def load_movies(file_path):
    """Load and clean movie names from a file."""
    try:
        # Load movies into a DataFrame
        movies = pd.read_csv(file_path, header=None, names=["Movie Name"])
        movies.dropna(inplace=True)  # Remove NaN rows
        movies["Movie Name"] = movies["Movie Name"].str.strip()  # Strip extra spaces
        return movies["Movie Name"].tolist()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []

def main():
    st.title("Movie Selector App")
    st.write("Upload a file with movie names and search for your favorite movies.")

    # File upload
    uploaded_file = st.file_uploader("Upload your movie file (CSV format with one column for movie names):")

    if uploaded_file:
        # Load and clean movie names
        movie_list = load_movies(uploaded_file)

        if movie_list:
            st.success(f"Successfully loaded {len(movie_list)} movies.")
            
            # Session state to track selected movies
            if "selected_movies" not in st.session_state:
                st.session_state.selected_movies = []

            # Limit selections to 5 movies
            if len(st.session_state.selected_movies) < 5:
                # Single search box for dynamic filtering
                query = st.text_input("Search movies:", key="search_input")

                # Filter the movie list based on the query
                if query and len(query) >= 3:
                    filtered_movies = [movie for movie in movie_list if query.lower() in movie.lower()]
                    
                    if filtered_movies:
                        selected_movie = st.selectbox(
                            "Select a movie:", 
                            filtered_movies, 
                            key="movie_select"
                        )
                        
                        # Add selected movie to the list if not already added
                        if selected_movie and selected_movie not in st.session_state.selected_movies:
                            if st.button("Add Movie"):
                                st.session_state.selected_movies.append(selected_movie)
                                st.success(f"Added: {selected_movie}")
                    else:
                        st.warning("No movies found matching your query.")
                else:
                    st.info("Type at least 3 characters to search.")
            else:
                st.warning("You have already selected 5 movies. No more can be added.")

            # Display selected movies
            if st.session_state.selected_movies:
                st.subheader("Selected Movies:")
                st.write(st.session_state.selected_movies)

        else:
            st.warning("No movies found in the file.")

if __name__ == "__main__":
    main()

