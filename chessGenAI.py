"""
Chess Piece Story Generator using LangChain
"""

import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables from .env file


class ChessStoryGenerator:
    """
    A class to generate stories about chess pieces using LangChain.
    """
    def __init__(self, api_key=None):
        """
        Initialize the story generator with the OpenAI API key.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will try to get from environment.
        """
        # Set the API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        # Initialize the OpenAI LLM
        self.llm = OpenAI(temperature=0.7)
        
        # Create memory for conversation history (optional for more contextual stories)
        self.memory = ConversationBufferMemory(input_key="piece", memory_key="chat_history")
        
        # Define the story templates for each piece
        self.piece_templates = {
            'pawn': "Write a kid-friendly story about a brave little pawn chess piece who dreams of reaching the other side of the board to become a queen. The story should be engaging, teach about how pawns move in chess, and include a positive moral lesson.",
            
            'knight': "Write a kid-friendly story about a knight chess piece who can jump over other pieces with its special L-shaped move. The story should be engaging, teach about how knights move in chess, and include a positive moral lesson.",
            
            'bishop': "Write a kid-friendly story about a bishop chess piece that can only move diagonally and always stays on the same color squares. The story should be engaging, teach about how bishops move in chess, and include a positive moral lesson.",
            
            'rook': "Write a kid-friendly story about a rook (castle) chess piece that can move horizontally and vertically. The story should be engaging, teach about how rooks move in chess, and include a positive moral lesson.",
            
            'queen': "Write a kid-friendly story about a powerful queen chess piece that can move in any direction (horizontally, vertically, and diagonally). The story should be engaging, teach about how queens move in chess, and include a positive moral lesson.",
            
            'king': "Write a kid-friendly story about a king chess piece that is the most important piece but can only move one square at a time in any direction. The story should be engaging, teach about how kings move in chess, and include a positive moral lesson."
        }
        
        # Create the prompt template
        self.template = """
        You are a creative storyteller for children who loves chess.
        
        {specific_prompt}
        
        The story should be about 300-400 words, have a clear beginning, middle, and end, use simple language appropriate for children ages 6-10, and make chess concepts fun and easy to understand.
        
        Chat history: {chat_history}
        
        Story about the {piece}:
        """
        
        self.prompt = PromptTemplate(
            input_variables=["specific_prompt", "chat_history", "piece"],
            template=self.template
        )
        
        # Create the story generation chain
        self.story_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )
    
    def generate_story(self, piece_type):
        """
        Generate a story for the specified chess piece.
        
        Args:
            piece_type (str): The type of chess piece ('pawn', 'knight', 'bishop', 'rook', 'queen', 'king')
            
        Returns:
            str: A story about the specified chess piece
        """
        # Validate the piece type
        piece_type = piece_type.lower()
        if piece_type not in self.piece_templates:
            return f"Sorry, I don't have a story template for '{piece_type}'. Please choose from: pawn, knight, bishop, rook, queen, or king."
        
        # Get the specific prompt for this piece
        specific_prompt = self.piece_templates[piece_type]
        
        # Generate the story
        story = self.story_chain.run({
            "specific_prompt": specific_prompt,
            "piece": piece_type
        })
        
        return story

def main():
    """
    Main function to demonstrate the story generator.
    """
    # Create the story generator
    try:
        generator = ChessStoryGenerator()
        
        # Example usage
        piece = input("Enter a chess piece (pawn, knight, bishop, rook, queen, king): ").lower()
        
        # Generate and print the story
        story = generator.generate_story(piece)
        print("\n" + story)
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OpenAI API key as an environment variable named 'OPENAI_API_KEY'")

if __name__ == "__main__":
    main()