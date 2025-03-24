# Chess Learning App with Kaggle Integration
# Requirements: pip install streamlit python-chess stockfish pandas scikit-learn

import streamlit as st
import chess
import chess.svg
import random
import pandas as pd
from stockfish import Stockfish
import os
import io
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import re

# Initialize Stockfish (you'll need to install this separately)
# Adjust the path to where you have Stockfish installed
stockfish_path = "stockfish" # Change this to your Stockfish executable path
try:
    stockfish = Stockfish(path=stockfish_path)
    stockfish_available = True
except Exception:
    stockfish_available = False

# Simple chess knowledge base for kids
piece_info = {
    'pawn': {
        'name': 'Pawn',
        'moves': 'Pawns move forward one square, but capture diagonally. On their first move, they can move two squares!',
        'value': 1,
        'fun_fact': 'Pawns are like brave little soldiers marching forward. They can transform into any other piece if they reach the other side of the board!'
    },
    'knight': {
        'name': 'Knight',
        'moves': 'Knights move in an L-shape: two squares in one direction and then one square perpendicular to that direction.',
        'value': 3,
        'fun_fact': 'Knights are the only pieces that can jump over other pieces. Their move looks like the letter "L"!'
    },
    'bishop': {
        'name': 'Bishop',
        'moves': 'Bishops move diagonally any number of squares.',
        'value': 3,
        'fun_fact': 'Bishops always stay on the same color squares they start on - either the light squares or the dark squares!'
    },
    'rook': {
        'name': 'Rook',
        'moves': 'Rooks move horizontally or vertically any number of squares.',
        'value': 5,
        'fun_fact': 'Rooks look like castles! They are very powerful pieces, especially when they control open files.'
    },
    'queen': {
        'name': 'Queen',
        'moves': 'Queens can move horizontally, vertically, or diagonally any number of squares.',
        'value': 9,
        'fun_fact': 'The queen is the most powerful piece on the board. She can move like both a rook and a bishop!'
    },
    'king': {
        'name': 'King',
        'moves': 'Kings move one square in any direction: horizontally, vertically, or diagonally.',
        'value': 'Priceless',
        'fun_fact': 'The king is the most important piece - if your king is captured (checkmate), you lose the game!'
    }
}

# Simple chess puzzles for beginners
beginner_puzzles = [
    {
        'fen': '8/8/8/4k3/8/8/3Q4/4K3 w - - 0 1',
        'task': 'Checkmate in 1 move',
        'solution': 'Qd5',
        'explanation': 'The queen moves to d5, putting the king in check. The king has no legal moves, so it\'s checkmate!'
    },
    {
        'fen': '8/8/8/3k4/8/8/3R4/4K3 w - - 0 1',
        'task': 'Checkmate in 1 move',
        'solution': 'Rd5',
        'explanation': 'The rook moves to d5, putting the king in check. The king has no legal moves, so it\'s checkmate!'
    },
    {
        'fen': 'k7/8/1R6/8/8/8/8/K7 w - - 0 1',
        'task': 'Checkmate in 1 move',
        'solution': 'Ra6',
        'explanation': 'The rook moves to a6, putting the king in check. The king can\'t move to b8 because the rook controls that square, so it\'s checkmate!'
    },
]

# Simple opening moves with explanations
simple_openings = [
    {
        'name': 'King\'s Pawn Opening',
        'move': 'e4',
        'explanation': 'This opens up the center of the board and allows your bishop and queen to move out.'
    },
    {
        'name': 'Queen\'s Pawn Opening',
        'move': 'd4',
        'explanation': 'Controls the center and opens a path for your bishop.'
    },
    {
        'name': 'Knight Opening',
        'move': 'Nf3',
        'explanation': 'Develops a knight and controls the center squares.'
    }
]

# ========== KAGGLE DATA INTEGRATION ==========

def download_and_prepare_kaggle_data():
    """
    Download and prepare chess datasets from Kaggle.
    In a real app, you would use the Kaggle API, but this example uses direct URLs for simplicity.
    """
    st.markdown("### Loading Chess Data...")
    
    # For demonstration, let's load a chess games dataset
    # In a real app, you would use:
    # kaggle.api.dataset_download_files('dataset_name', path='./data', unzip=True)
    
    # For this example, we'll use a direct download link (replace with actual Kaggle dataset URL)
    try:
        # Example: Loading a dataset of beginner games
        # Note: In a real app, use the Kaggle API instead of direct URLs
        url = "https://example.com/chess_games_for_beginners.csv"  # Replace with actual URL
        
        # Simulated data for demonstration
        # In a real app, you would load this from the downloaded CSV
        sample_data = """game_id,white_rating,black_rating,opening_name,opening_moves,result,num_moves,white_blunders,black_blunders
1,1200,1150,King's Pawn Opening,e4 e5,1-0,25,2,4
2,1100,1300,Queen's Gambit,d4 d5 c4,0-1,30,3,1
3,1250,1200,Sicilian Defense,e4 c5,1/2-1/2,45,1,2
4,1400,1350,Ruy Lopez,e4 e5 Nf3 Nc6 Bb5,1-0,35,1,3
5,1180,1220,Italian Game,e4 e5 Nf3 Nc6 Bc4,0-1,28,3,2"""
        
        # In a real app, you would use:
        # games_df = pd.read_csv(url)
        games_df = pd.read_csv(io.StringIO(sample_data))
        
        st.success(f"Successfully loaded chess games data with {len(games_df)} records")
        return games_df
    except Exception as e:
        st.error(f"Error loading chess data: {str(e)}")
        # Return a minimal sample dataset for demonstration
        return pd.DataFrame({
            'game_id': range(1, 6),
            'white_rating': [1200, 1100, 1250, 1400, 1180],
            'black_rating': [1150, 1300, 1200, 1350, 1220],
            'opening_name': ['King\'s Pawn Opening', 'Queen\'s Gambit', 'Sicilian Defense', 'Ruy Lopez', 'Italian Game'],
            'opening_moves': ['e4 e5', 'd4 d5 c4', 'e4 c5', 'e4 e5 Nf3 Nc6 Bb5', 'e4 e5 Nf3 Nc6 Bc4'],
            'result': ['1-0', '0-1', '1/2-1/2', '1-0', '0-1'],
            'num_moves': [25, 30, 45, 35, 28],
            'white_blunders': [2, 3, 1, 1, 3],
            'black_blunders': [4, 1, 2, 3, 2]
        })

def load_puzzle_data():
    """
    Load chess puzzles from Kaggle dataset.
    """
    try:
        # Simulated puzzles data (would be loaded from Kaggle in a real app)
        sample_data = """puzzle_id,fen,moves,rating,themes
1,r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3,Bb5,800,development
2,r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4,O-O,900,castling
3,rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2,Nf3,700,development
4,rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2,Nf3,700,development
5,r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4,O-O,900,castling"""
        
        puzzles_df = pd.read_csv(io.StringIO(sample_data))
        return puzzles_df
    except Exception as e:
        st.error(f"Error loading puzzle data: {str(e)}")
        return pd.DataFrame({
            'puzzle_id': range(1, 6),
            'fen': [
                'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
                'r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4',
                'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2',
                'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2',
                'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4'
            ],
            'moves': ['Bb5', 'O-O', 'Nf3', 'Nf3', 'O-O'],
            'rating': [800, 900, 700, 700, 900],
            'themes': ['development', 'castling', 'development', 'development', 'castling']
        })

def get_kid_friendly_puzzles(puzzles_df, difficulty_level=1):
    """
    Filter and adapt puzzles to be kid-friendly based on difficulty level.
    """
    # Filter puzzles by rating (lower = easier)
    if difficulty_level == 1:  # Beginner
        filtered_puzzles = puzzles_df[puzzles_df['rating'] < 800]
    elif difficulty_level == 2:  # Intermediate
        filtered_puzzles = puzzles_df[(puzzles_df['rating'] >= 800) & (puzzles_df['rating'] < 1000)]
    else:  # Advanced
        filtered_puzzles = puzzles_df[puzzles_df['rating'] >= 1000]
    
    # If we don't have enough puzzles after filtering, take the easiest ones available
    if len(filtered_puzzles) < 5:
        filtered_puzzles = puzzles_df.sort_values('rating').head(5)
    
    # Convert to our puzzle format
    kid_puzzles = []
    for _, puzzle in filtered_puzzles.iterrows():
        # Create a kid-friendly explanation based on the theme
        if puzzle['themes'] == 'development':
            explanation = "This move helps develop your pieces and control the center!"
        elif puzzle['themes'] == 'castling':
            explanation = "Castling keeps your king safe and connects your rooks!"
        else:
            explanation = "This is a good move because it improves your position!"
        
        kid_puzzles.append({
            'fen': puzzle['fen'],
            'task': f"Find the best move! (Hint: Think about {puzzle['themes']})",
            'solution': puzzle['moves'],
            'explanation': explanation
        })
    
    return kid_puzzles

def extract_openings_from_games(games_df):
    """
    Extract common openings from games dataset and create kid-friendly explanations.
    """
    # Get the most common openings
    common_openings = games_df['opening_name'].value_counts().head(10).index.tolist()
    
    # Create a dictionary of openings with their moves and explanations
    openings_dict = {}
    for opening in common_openings:
        # Get the first game with this opening
        game = games_df[games_df['opening_name'] == opening].iloc[0]
        
        # Extract first move
        first_move = game['opening_moves'].split()[0]
        
        # Create a kid-friendly explanation
        if 'Pawn' in opening or first_move in ['e4', 'd4', 'c4']:
            explanation = "This opens up the center and helps your bishops and queen move out!"
        elif 'Knight' in opening or first_move in ['Nf3', 'Nc3']:
            explanation = "Developing your knight controls important squares in the center!"
        else:
            explanation = "This is a popular opening that many strong players use!"
        
        openings_dict[opening] = {
            'name': opening,
            'move': first_move,
            'explanation': explanation
        }
    
    # Convert to list for our app format
    kid_openings = list(openings_dict.values())
    return kid_openings

def train_simple_mistake_detector(games_df):
    """
    Train a simple model to detect common beginner mistakes.
    This is a simplified example - a real implementation would use more features.
    """
    # Create very simple features (in a real app, you would extract more meaningful features)
    # For demonstration only - this is not how you'd really build a chess mistake detector
    X = games_df[['white_rating', 'black_rating', 'num_moves']]
    
    # Target: did white make more blunders than black?
    y = (games_df['white_blunders'] > games_df['black_blunders']).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create a simple function to generate advice
    def get_advice_for_position(board, player_rating=1000):
        # In a real app, you would extract meaningful features from the board position
        # This is just a placeholder implementation
        features = np.array([[
            player_rating,  # player rating
            1200,  # opponent rating (fixed for demo)
            board.fullmove_number  # current move number
        ]])
        
        # Predict if the player might make a mistake
        prediction = model.predict(features)[0]
        
        if prediction == 1:
            # More likely to make a mistake
            return "Be careful! Look for checks, captures, and threats before moving."
        else:
            return "Keep developing your pieces and controlling the center."
    
    return get_advice_for_position

# ========== MAIN APP CODE ==========

# App title
st.title("Chess Buddy - Learn Chess for Kids!")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Chess_Pieces_Sprite.svg/800px-Chess_Pieces_Sprite.svg.png", width=200)

# Initialize session state variables
if 'board' not in st.session_state:
    st.session_state.board = chess.Board()
    
if 'puzzle_solved' not in st.session_state:
    st.session_state.puzzle_solved = False
    
if 'current_puzzle' not in st.session_state:
    st.session_state.current_puzzle = 0
    
if 'show_solution' not in st.session_state:
    st.session_state.show_solution = False
    
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.games_df = None
    st.session_state.puzzles_df = None
    st.session_state.mistake_detector = None

# Kaggle data loading section in sidebar
st.sidebar.markdown("---")
st.sidebar.header("Data and AI Settings")

if not st.session_state.data_loaded:
    if st.sidebar.button("Load Chess Data"):
        # Load game data
        st.session_state.games_df = download_and_prepare_kaggle_data()
        
        # Load puzzle data
        st.session_state.puzzles_df = load_puzzle_data()
        
        # Extract kid-friendly openings
        if st.session_state.games_df is not None:
            st.session_state.kid_openings = extract_openings_from_games(st.session_state.games_df)
            
        # Get kid-friendly puzzles
        if st.session_state.puzzles_df is not None:
            difficulty = 1  # Default to beginner
            st.session_state.kid_puzzles = get_kid_friendly_puzzles(st.session_state.puzzles_df, difficulty)
            
        # Train simple mistake detector
        if st.session_state.games_df is not None:
            st.session_state.mistake_detector = train_simple_mistake_detector(st.session_state.games_df)
            
        st.session_state.data_loaded = True

# App sections
app_mode_options = ["Meet the Chess Pieces", "Learn How Pieces Move", "Try a Puzzle", "Play Against Computer"]
if st.session_state.data_loaded:
    app_mode_options.append("Chess Analytics")

app_mode = st.sidebar.selectbox(
    "Choose what you want to learn:",
    app_mode_options
)

# SECTION 1: MEET THE CHESS PIECES
if app_mode == "Meet the Chess Pieces":
    st.header("Meet the Chess Pieces")
    
    selected_piece = st.selectbox(
        "Select a piece to learn about:",
        list(piece_info.keys())
    )
    
    piece = piece_info[selected_piece]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # We would use a real image here
        st.markdown(f"# {piece['name']}")
        st.markdown(f"**Value:** {piece['value']} points")
    
    with col2:
        st.markdown(f"### How it moves:")
        st.markdown(piece['moves'])
        st.markdown("### Fun Fact:")
        st.markdown(piece['fun_fact'])

# SECTION 2: LEARN HOW PIECES MOVE
elif app_mode == "Learn How Pieces Move":
    st.header("Learn How Pieces Move")
    
    selected_piece = st.selectbox(
        "Select a piece to see how it moves:",
        list(piece_info.keys())
    )
    
    # Create a clean board to demonstrate moves
    demo_board = chess.Board(fen="8/8/8/8/8/8/8/8 w - - 0 1")
    
    # Place the selected piece in the center
    if selected_piece == 'pawn':
        demo_board.set_piece_at(chess.E4, chess.Piece.from_symbol('P'))
        legal_moves = [chess.E5, chess.E6]  # Simplified for demo
    elif selected_piece == 'knight':
        demo_board.set_piece_at(chess.E4, chess.Piece.from_symbol('N'))
        legal_moves = [chess.D6, chess.F6, chess.C5, chess.G5, chess.C3, chess.G3, chess.D2, chess.F2]
    elif selected_piece == 'bishop':
        demo_board.set_piece_at(chess.E4, chess.Piece.from_symbol('B'))
        legal_moves = [chess.D3, chess.C2, chess.B1, chess.F5, chess.G6, chess.H7, chess.D5, chess.C6, chess.B7, chess.A8, chess.F3, chess.G2, chess.H1]
    elif selected_piece == 'rook':
        demo_board.set_piece_at(chess.E4, chess.Piece.from_symbol('R'))
        legal_moves = [chess.E1, chess.E2, chess.E3, chess.E5, chess.E6, chess.E7, chess.E8, chess.A4, chess.B4, chess.C4, chess.D4, chess.F4, chess.G4, chess.H4]
    elif selected_piece == 'queen':
        demo_board.set_piece_at(chess.E4, chess.Piece.from_symbol('Q'))
        legal_moves = [
            chess.E1, chess.E2, chess.E3, chess.E5, chess.E6, chess.E7, chess.E8,  # Vertical
            chess.A4, chess.B4, chess.C4, chess.D4, chess.F4, chess.G4, chess.H4,  # Horizontal
            chess.D3, chess.C2, chess.B1, chess.F5, chess.G6, chess.H7,  # Diagonal 1
            chess.D5, chess.C6, chess.B7, chess.A8, chess.F3, chess.G2, chess.H1  # Diagonal 2
        ]
    elif selected_piece == 'king':
        demo_board.set_piece_at(chess.E4, chess.Piece.from_symbol('K'))
        legal_moves = [chess.D3, chess.E3, chess.F3, chess.D4, chess.F4, chess.D5, chess.E5, chess.F5]
    
    # Get SVG for board with highlighted moves
    squares = [chess.parse_square(square) for square in [
        move.uci()[2:4] for move in demo_board.legal_moves
    ]] if selected_piece != 'pawn' else legal_moves
    
    highlighted_squares = {square: "#rrggbb" for square in squares}
    if highlighted_squares:
        highlighted_squares[chess.E4] = "#00FF00"  # Highlight the piece position in green
    
    board_svg = chess.svg.board(
        demo_board, 
        size=400,
        fill=highlighted_squares,
        arrows=[chess.svg.Arrow(chess.E4, move, color="#0000FF") for move in squares]
    )
    
    st.markdown(f"### {piece_info[selected_piece]['name']} Moves")
    st.markdown(piece_info[selected_piece]['moves'])
    st.components.v1.html(board_svg, height=410)
    
    st.markdown("The blue arrows show all possible moves for this piece from its current position.")

# SECTION 3: TRY A PUZZLE
elif app_mode == "Try a Puzzle":
    st.header("Chess Puzzles for Beginners")
    
    if st.session_state.data_loaded and hasattr(st.session_state, 'kid_puzzles') and st.session_state.kid_puzzles:
        # Difficulty selection
        difficulty = st.slider("Puzzle Difficulty", 1, 3, 1, 
                              help="1=Beginner, 2=Intermediate, 3=Advanced")
        
        # Update puzzles if difficulty changes
        if 'current_difficulty' not in st.session_state or st.session_state.current_difficulty != difficulty:
            st.session_state.kid_puzzles = get_kid_friendly_puzzles(st.session_state.puzzles_df, difficulty)
            st.session_state.current_difficulty = difficulty
            st.session_state.current_puzzle = 0
            st.session_state.puzzle_refreshed = True
        
        # Get current puzzle
        puzzle_idx = st.session_state.current_puzzle
        puzzles_to_use = st.session_state.kid_puzzles
    else:
        # Use the default puzzles if Kaggle data not loaded
        puzzle_idx = st.session_state.current_puzzle
        puzzles_to_use = beginner_puzzles
    
    # Get the current puzzle
    if len(puzzles_to_use) > 0:
        puzzle = puzzles_to_use[puzzle_idx % len(puzzles_to_use)]
        
        # Display puzzle
        st.markdown(f"### Puzzle {puzzle_idx + 1}: {puzzle['task']}")
        
        # Set up board with the puzzle position
        if 'puzzle_board' not in st.session_state or st.session_state.puzzle_refreshed:
            st.session_state.puzzle_board = chess.Board(puzzle['fen'])
            st.session_state.puzzle_refreshed = False
        
        # Display board
        board_svg = chess.svg.board(st.session_state.puzzle_board, size=400)
        st.components.v1.html(board_svg, height=410)
        
        # User input for the solution
        user_move = st.text_input("Enter your move (e.g., 'e2e4' or 'Nf3'):", key="puzzle_move")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Check Solution"):
                solution_san = puzzle['solution']
                solution_uci = None
                
                # Convert SAN to UCI if needed
                for move in st.session_state.puzzle_board.legal_moves:
                    if st.session_state.puzzle_board.san(move) == solution_san:
                        solution_uci = move.uci()
                        break
                
                if user_move.lower() == solution_san.lower() or user_move.lower() == solution_uci:
                    st.session_state.puzzle_solved = True
                    st.success("Correct! Great job!")
                    st.balloons()
                else:
                    st.error("Not quite right. Try again!")
                    
        with col2:
            if st.button("Show Solution"):
                st.session_state.show_solution = True
        
        with col3:
            if st.button("Next Puzzle"):
                st.session_state.current_puzzle = (st.session_state.current_puzzle + 1) % len(puzzles_to_use)
                st.session_state.puzzle_board = chess.Board(puzzles_to_use[st.session_state.current_puzzle]['fen'])
                st.session_state.puzzle_solved = False
                st.session_state.show_solution = False
                st.experimental_rerun()
        
        # Show solution if requested
        if st.session_state.show_solution or st.session_state.puzzle_solved:
            st.markdown(f"**Solution:** {puzzle['solution']}")
            st.markdown(f"**Explanation:** {puzzle['explanation']}")

# SECTION 4: PLAY AGAINST COMPUTER
elif app_mode == "Play Against Computer":
    st.header("Play Against the Computer")
    
    if not stockfish_available:
        st.warning("Stockfish chess engine not found. Please install Stockfish to enable this feature.")
        st.markdown("""
        To install Stockfish:
        1. Download it from [Stockfish website](https://stockfishchess.org/download/)
        2. Update the stockfish_path variable in the code
        """)
    else:
        # Difficulty level
        difficulty = st.sidebar.slider("Computer Difficulty (1=Easiest, 10=Hardest)", 1, 10, 1)
        stockfish.set_skill_level(difficulty)
        
        # Reset button
        if st.sidebar.button("Start New Game"):
            st.session_state.board = chess.Board()
            st.session_state.last_move = None
        
        # Display current board state
        last_move = None if 'last_move' not in st.session_state else st.session_state.last_move
        arrows = [] if last_move is None else [chess.svg.Arrow(last_move[0], last_move[1], color="#0000FF")]
        
        board_svg = chess.svg.board(st.session_state.board, size=400, arrows=arrows)
        st.components.v1.html(board_svg, height=410)
        
        # Add AI advice if data is loaded
        if st.session_state.data_loaded and st.session_state.mistake_detector:
            if not st.session_state.board.is_game_over():
                # Get AI advice
                player_rating = 1000  # Default rating, could be adjustable
                advice = st.session_state.mistake_detector(st.session_state.board, player_rating)
                
                # Display advice
                st.info(f"AI Coach says: {advice}")
        
        # Show board status
        if st.session_state.board.is_checkmate():
            st.markdown("### Checkmate! Game over.")
            if st.session_state.board.turn:
                st.markdown("Black wins!")
            else:
                st.markdown("White wins!")
        elif st.session_state.board.is_stalemate():
            st.markdown("### Stalemate! Game over.")
        elif st.session_state.board.is_check():
            st.markdown("### Check!")
        
        if not st.session_state.board.is_game_over():
            # Player's move input
            col1, col2 = st.columns(2)
            
            with col1:
                from_square = st.text_input("From square (e.g., e2):", key="from_square")
            
            with col2:
                to_square = st.text_input("To square (e.g., e4):", key="to_square")

            if st.button("Make Move"):
                if from_square and to_square:
                    try:
                        from_sq = chess.parse_square(from_square.lower())
                        to_sq = chess.parse_square(to_square.lower())
                        # ... (your chess move logic here, using from_sq and to_sq) ...
                    except ValueError as ve: #Catching the specific value error that chess.parse_square can raise
                        st.error(f"Invalid square input: {ve}") #st.error is better than return in streamlit.
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}") #st.error is better than return in streamlit.
                    else:
                        # Code to execute if no exceptions occurred.
                        st.success("Move made successfully!")
                        # ... (update game state, display updated board, etc.) ...
                else:
                    st.warning("Please select both 'From' and 'To' squares.") #st.warning is better than return in streamlit.

            
          #  if st.button("Make Move"):
          #      if from_square and to_square:
          #         try:
          #              from_sq = chess.parse_square(from_square.lower())
           #             to_sq = chess.parse_square(to_square.lower())
            #        except Exception as e:
             #           return f"An error occurred: {e}"
                        
                        # Create the move
                