"""
Integration module to connect the Chess Learning App with the ChessStoryGenerator
"""

import os
import streamlit as st
from chessGenAI import ChessStoryGenerator

# This function initializes the story generator if it doesn't exist in session state
def initialize_story_generator():
    """
    Initialize the ChessStoryGenerator and store it in Streamlit's session state
    """
    if 'story_generator' not in st.session_state:
        try:
            # Try to create the story generator
            # If you have an API key stored as an environment variable, it will use that
            st.session_state.story_generator = ChessStoryGenerator()
            st.session_state.story_generator_available = True
        except ValueError:
            # If no API key is available, we'll set a flag to show an appropriate message
            st.session_state.story_generator_available = False
            st.error("Story generator requires an OpenAI API key. Please set the OPENAI_API_KEY environment variable.")

# This function gets a story for a specific chess piece
def get_chess_piece_story(piece_type):
    """
    Get a story for the specified chess piece using the ChessStoryGenerator
    
    Args:
        piece_type (str): The type of chess piece ('pawn', 'knight', 'bishop', 'rook', 'queen', 'king')
        
    Returns:
        str: A story about the specified chess piece or an error message
    """
    # Initialize the generator if it's not already done
    initialize_story_generator()
    
    # Check if the generator is available
    if not st.session_state.story_generator_available:
        return "Story generator is not available. Please check that you've set up your OpenAI API key properly."
    
    # Generate the story
    try:
        with st.spinner(f"Generating a story about the {piece_type.capitalize()}..."):
            story = st.session_state.story_generator.generate_story(piece_type)
        return story
    except Exception as e:
        return f"An error occurred while generating the story: {str(e)}"

# This function can be used to integrate the story generator into the Meet the Chess Pieces section
def add_story_button_to_chess_app(selected_piece):
    """
    Add a story button to the chess app for the selected piece
    
    Args:
        selected_piece (str): The type of chess piece selected in the app
        
    Returns:
        None: This function adds Streamlit UI elements directly
    """
    # Add a divider
    st.markdown("---")
    st.subheader(f"Story of the {selected_piece.capitalize()}")
    
    # Check if API key is configured
    api_key_configured = os.environ.get("OPENAI_API_KEY") is not None
    
    if not api_key_configured:
        st.warning("To enable AI-generated stories, please set your OpenAI API key in the environment variables.")
        
        # Option to enter API key in the app (not secure for production)
        with st.expander("Configure API Key (Development Only)"):
            api_key = st.text_input("Enter your OpenAI API key:", type="password")
            if api_key and st.button("Save API Key"):
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("API key saved for this session!")
                st.session_state.story_generator_available = None  # Reset so it will be re-initialized
    
    # Create the button to generate a story
    if st.button("Tell me a story about this piece!"):
        if not api_key_configured and 'story_generator_available' not in st.session_state:
            st.warning("Please configure your OpenAI API key first.")
        else:
            story = get_chess_piece_story(selected_piece)
            st.markdown(story)
            st.balloons()

# Example of how to implement this in the main app:
"""
# In the "Meet the Chess Pieces" section of your main app, add:

from chess_app_integration import add_story_button_to_chess_app

# After displaying the piece info
add_story_button_to_chess_app(selected_piece)
"""