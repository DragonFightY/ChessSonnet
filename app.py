# app.py
from flask import Flask, render_template, jsonify, request
from chessgpt import ChessGPT
import chess
import logging
import os

# Set up logging with your original configuration
logging.basicConfig(
    filename='chessGPT_web.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Initialize ChessGPT with web-specific configurations
chess_gpt = ChessGPT(use_stockfish=False)

# Dictionary to store active games
active_games = {}

@app.route('/')
def index():
    """Render the main ChessGPT web interface."""
    return render_template('index.html')

@app.route('/new_game', methods=['POST'])
def new_game():
    """Initialize a new game and return its ID."""
    game_id = str(len(active_games))
    active_games[game_id] = {
        'board': chess.Board(),
        'moves': []
    }
    return jsonify({
        'game_id': game_id,
        'fen': active_games[game_id]['board'].fen()
    })

@app.route('/make_move', methods=['POST'])
def make_move():
    """Process a player's move and respond with AI's move."""
    data = request.get_json()
    game_id = data.get('game_id')
    move_uci = data.get('move')
    
    if game_id not in active_games:
        return jsonify({'error': 'Invalid game ID'}), 400
        
    game = active_games[game_id]
    board = game['board']
    
    try:
        # Process player's move
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return jsonify({'error': 'Illegal move'}), 400
            
        board.push(move)
        game['moves'].append(move_uci)
        
        # Get and make AI's move
        if not board.is_game_over():
            ai_move = chess_gpt.get_best_move(board)
            if ai_move:
                board.push(ai_move)
                game['moves'].append(ai_move.uci())
        
        # Store position for learning
        chess_gpt.store_move(board)
        
        # If game is over, learn from it
        if board.is_game_over():
            final_reward = 1 if board.is_checkmate() else 0
            chess_gpt.learn_from_game(final_reward, len(game['moves']))
        
        return jsonify({
            'fen': board.fen(),
            'game_over': board.is_game_over(),
            'in_check': board.is_check(),
            'result': board.result() if board.is_game_over() else None,
            'last_move': ai_move.uci() if not board.is_game_over() and ai_move else None
        })
        
    except Exception as e:
        logging.error(f"Move error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    """Return current training metrics."""
    return jsonify(chess_gpt.metrics)

if __name__ == '__main__':
    app.run(debug=True)

# chessgpt.py (Your original AI code with web adaptations)
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.engine
import numpy as np
import os
import json
import logging
from threading import Lock

class ImprovedChessNet(nn.Module):
    """Your original neural network architecture."""
    def __init__(self):
        super(ImprovedChessNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.tanh(x)

class ChessGPT:
    """Your ChessGPT class adapted for web use."""
    def __init__(self, use_stockfish=False, stockfish_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ImprovedChessNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.game_history = []
        self.lock = Lock()  # Add thread safety for web environment
        
        # Initialize metrics
        self.metrics = {
            'total_games': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'average_game_length': 0,
            'total_moves': 0,
            'running_loss': []
        }
        
        self.load_training_state()
        self.load_metrics()
        
        # Initialize Stockfish if requested
        self.use_stockfish = use_stockfish
        self.stockfish_path = stockfish_path
        self.engine = None
        if use_stockfish and stockfish_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            except Exception as e:
                logging.error(f"Failed to initialize Stockfish: {e}")
                self.use_stockfish = False

    # Your existing methods (load_training_state, save_training_state, etc.)
    # Add the Lock usage where appropriate for thread safety

    def get_best_move(self, board):
        """Get the best move for the current position."""
        with self.lock:
            # Your existing get_best_move implementation
            pass

    def learn_from_game(self, final_reward, moves_made):
        """Learn from the completed game."""
        with self.lock:
            # Your existing learning implementation
            pass

    def __del__(self):
        """Clean up resources."""
        if self.engine:
            self.engine.quit()