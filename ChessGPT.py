import json
import tkinter as tk
from tkinter import messagebox, ttk
import chess
import chess.svg
import chess.engine
from PIL import Image, ImageTk
import cairosvg
import io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import threading
import time
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(
    filename='chessGPT_game.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
class ChessAI:
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
        
        self.load_training_state()
        self.load_metrics()
        self.model.eval()
        
        self.use_stockfish = use_stockfish
        self.stockfish_path = stockfish_path
        self.stockfish_depth = stockfish_depth
        self.engine = None
        if use_stockfish and stockfish_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                self.engine.configure({
                    "Threads": os.cpu_count(),
                    "Hash": 2048,
                    "Minimum Thinking Time": 20
                })
            except Exception as e:
                logging.error(f"Failed to initialize Stockfish: {e}")
                self.use_stockfish = False

    def __del__(self):
        if self.engine:
            self.engine.quit()

    def save_metrics(self):
        try:
            metrics_path = 'chessGPT_metrics.json'
            with open(metrics_path, 'w', buffering=8192) as f:
                json.dump(self.metrics, f)
            logging.info("Metrics saved successfully")
        except Exception as e:
            logging.error(f"Error saving metrics: {e}")

    def load_metrics(self):
        try:
            metrics_path = 'chessGPT_metrics.json'
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', buffering=8192) as f:
                    self.metrics = json.load(f)
                logging.info("Metrics loaded successfully")
        except Exception as e:
            logging.error(f"Error loading metrics: {e}")

    def save_training_state(self, force=False):
        try:
            save_path = 'chessGPT_model.pth'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_games': self.training_games
            }, save_path, _use_new_zipfile_serialization=True)
            logging.info(f"AI model saved successfully after {self.training_games} training games")
            return True
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            return False

    def load_training_state(self):
        try:
            if os.path.exists('chessGPT_model.pth'):
                checkpoint = torch.load('chessGPT_model.pth', map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_games = checkpoint.get('training_games', 0)
                logging.info(f"AI model loaded successfully with {self.training_games} training games")
                return True
            return False
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False

    def board_to_tensor(self, board):
        pieces = ['p', 'n', 'b', 'r', 'q', 'k']
        tensor = np.zeros((6, 8, 8), dtype=np.float32)
        
        for i in range(64):
            piece = board.piece_at(i)
            if piece is not None:
                piece_type = pieces.index(piece.symbol().lower())
                rank, file = i // 8, i % 8
                tensor[piece_type][rank][file] = 1 if piece.color else -1
        
        return torch.FloatTensor(tensor).unsqueeze(0).to(self.device)

    def get_stockfish_move(self, board, time_limit=1.0):
        try:
            result = self.engine.play(
                board,
                chess.engine.Limit(time=time_limit, depth=self.stockfish_depth)
            )
            return result.move
        except Exception as e:
            logging.error(f"Stockfish error: {e}")
            return None

    def evaluate_position(self, board):
        if self.use_stockfish and self.engine:
            try:
                info = self.engine.analyse(board, chess.engine.Limit(depth=self.stockfish_depth))
                return info.get("score", chess.engine.Score(0)).white().score()
            except:
                pass

        if board.is_checkmate():
            return -1000 if board.turn else 1000
        if board.is_stalemate():
            return 0
        
        with torch.no_grad():
            state = self.board_to_tensor(board)
            network_eval = self.model(state).item()

        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        material_score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                material_score += value if piece.color else -value

        return 0.7 * (material_score / 100) + 0.3 * network_eval

    def get_best_move(self, board, depth=3):
        if self.use_stockfish and self.engine:
            stockfish_move = self.get_stockfish_move(board)
            if stockfish_move:
                return stockfish_move

        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        legal_moves = list(board.legal_moves)
        random.shuffle(legal_moves)
        
        for move in legal_moves:
            board.push(move)
            value = -self.minimax(board, depth-1, -beta, -alpha)
            board.pop()
            
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
            
        return best_move

    def minimax(self, board, depth, alpha, beta):
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board)
        
        best_value = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            value = -self.minimax(board, depth-1, -beta, -alpha)
            board.pop()
            
            best_value = max(best_value, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        
        return best_value

    def learn_from_game(self, final_reward, moves_made):
        if not self.game_history:
            return
        
        self.model.train()
        total_loss = 0
        
        for i, state in enumerate(self.game_history):
            self.optimizer.zero_grad()
            prediction = self.model(state)
            reward = final_reward * (0.95 ** (len(self.game_history) - i))
            loss = nn.MSELoss()(prediction, torch.tensor([[reward]], device=self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.game_history)
        self.game_history = []
        self.model.eval()
        self.training_games += 1
        
        self.metrics['total_games'] += 1
        self.metrics['total_moves'] += moves_made
        self.metrics['average_game_length'] = self.metrics['total_moves'] / self.metrics['total_games']
        self.metrics['running_loss'].append(avg_loss)
        
        if final_reward > 0:
            self.metrics['wins'] += 1
        elif final_reward < 0:
            self.metrics['losses'] += 1
        else:
            self.metrics['draws'] += 1
            
        if len(self.metrics['running_loss']) > 1000:
            self.metrics['running_loss'] = self.metrics['running_loss'][-1000:]
        
        logging.info(f"""Training game {self.training_games} completed:
            Average loss: {avg_loss:.4f}
            Win rate: {(self.metrics['wins']/self.metrics['total_games'])*100:.2f}%
            Average game length: {self.metrics['average_game_length']:.1f} moves
        """)
        
        if self.training_games % 1 == 0:
            self.save_training_state()
            self.save_metrics()

    def store_move(self, board):
        state = self.board_to_tensor(board)
        self.game_history.append(state)

    def parallel_self_play(self, num_games_per_process):
        games_data = []
        for _ in range(num_games_per_process):
            board = chess.Board()
            moves_without_capture = 0
            moves_made = 0
            
            while not board.is_game_over() and moves_without_capture < 50:
                move = self.get_best_move(board)
                if move:
                    self.store_move(board)
                    moves_made += 1
                    
                    is_capture = board.is_capture(move)
                    is_pawn_move = board.piece_at(move.from_square).piece_type == chess.PAWN
                    
                    board.push(move)
                    
                    if is_capture or is_pawn_move:
                        moves_without_capture = 0
                    else:
                        moves_without_capture += 1
                else:
                    break
            
            final_reward = 1 if board.is_checkmate() and not board.turn else (-1 if board.is_checkmate() else 0)
            games_data.append((final_reward, moves_made))
        
        return games_data

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess AI Interface")
        
        self.board = chess.Board()
        self.selected_square = None
        self.player_color = chess.WHITE
        
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)
        
        self.canvas_size = 400
        self.square_size = self.canvas_size // 8
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(side=tk.LEFT)
        
        self.side_panel = tk.Frame(self.main_frame)
        self.side_panel.pack(side=tk.LEFT, padx=20)
        
        self.ai_frame = tk.LabelFrame(self.side_panel, text="AI Settings", padx=5, pady=5)
        self.ai_frame.pack(pady=10, fill="x")
        
        self.ai_type = tk.StringVar(value="neural")
        tk.Radiobutton(self.ai_frame, text="Neural Network", variable=self.ai_type, 
                      value="neural", command=self.on_ai_type_change).pack()
        tk.Radiobutton(self.ai_frame, text="Stockfish", variable=self.ai_type,
                      value="stockfish", command=self.on_ai_type_change).pack()
        
        self.stockfish_frame = tk.Frame(self.ai_frame)
        self.stockfish_frame.pack(pady=5, fill="x")
        tk.Label(self.stockfish_frame, text="Stockfish path:").pack()
        self.stockfish_path = tk.StringVar(value="stockfish")
        tk.Entry(self.stockfish_frame, textvariable=self.stockfish_path).pack()
        
        self.metrics_frame = tk.LabelFrame(self.side_panel, text="Training Metrics", padx=5, pady=5)
        self.metrics_frame.pack(pady=10, fill="x")
        
        self.metrics_text = tk.Text(self.metrics_frame, height=6, width=30)
        self.metrics_text.pack(pady=5)
        
        tk.Button(self.metrics_frame, text="Update Metrics", 
                 command=self.update_metrics_display).pack()
        
        tk.Button(self.side_panel, text="New Game", command=self.new_game).pack(pady=5)
        tk.Button(self.side_panel, text="AI Move", command=self.make_ai_move).pack(pady=5)
        
        self.training_frame = tk.LabelFrame(self.side_panel, text="Self-Play Training", padx=5,pady=5)
        self.training_frame.pack(pady=10, fill="x")
        
        self.num_games_var = tk.StringVar(value="100")
        tk.Label(self.training_frame, text="Number of games:").pack()
        tk.Entry(self.training_frame, textvariable=self.num_games_var).pack()
        
        self.progress_var = tk.StringVar(value="Training progress: 0%")
        self.progress_label = tk.Label(self.training_frame, textvariable=self.progress_var)
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(self.training_frame, length=200, mode='determinate')
        self.progress_bar.pack(pady=5)
        
        self.train_button = tk.Button(self.training_frame, text="Start Training", 
                                    command=self.start_self_play_training)
        self.train_button.pack()
        
        self.status_var = tk.StringVar(value="White to move")
        self.status_label = tk.Label(self.side_panel, textvariable=self.status_var)
        self.status_label.pack(pady=10)
        
        self.initialize_ai()
        
        self.canvas.bind('<Button-1>', self.on_square_click)
        
        self.update_board()
        self.update_metrics_display()

    def update_metrics_display(self):
        if hasattr(self.chess_ai, 'metrics'):
            metrics = self.chess_ai.metrics
            total_games = metrics['total_games']
            win_rate = (metrics['wins']/total_games*100) if total_games > 0 else 0
            
            metrics_text = f"""Total Games: {total_games}
Win Rate: {win_rate:.2f}%
Average Game Length: {metrics['average_game_length']:.1f}
Recent Avg Loss: {np.mean(metrics['running_loss'][-100:]) if metrics['running_loss'] else 0:.4f}
Wins: {metrics['wins']}
Draws: {metrics['draws']}
Losses: {metrics['losses']}"""
            
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, metrics_text)

    def initialize_ai(self):
        try:
            use_stockfish = self.ai_type.get() == "stockfish"
            stockfish_path = self.stockfish_path.get() if use_stockfish else None
            self.chess_ai = ChessAI(use_stockfish=use_stockfish, 
                                  stockfish_path=stockfish_path)
            self.has_ai = True
            self.training_in_progress = False
        except Exception as e:
            self.has_ai = False
            messagebox.showwarning("Warning", f"AI initialization error: {str(e)}")

    def on_ai_type_change(self):
        self.initialize_ai()

    def start_self_play_training(self):
        if self.training_in_progress or self.ai_type.get() == "stockfish":
            return
        
        try:
            num_games = int(self.num_games_var.get())
            if num_games <= 0:
                raise ValueError("Number of games must be positive")
            
            self.train_button.config(state='disabled')
            self.training_in_progress = True
            self.progress_bar['maximum'] = num_games
            
            training_thread = threading.Thread(target=self.self_play_training, 
                                            args=(num_games,))
            training_thread.daemon = True
            training_thread.start()
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def self_play_training(self, num_games):
        num_processes = os.cpu_count()
        games_per_process = max(1, num_games // num_processes)
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(self.chess_ai.parallel_self_play, games_per_process) 
                      for _ in range(num_processes)]
            
            completed_games = 0
            for future in futures:
                games_data = future.result()
                for final_reward, moves_made in games_data:
                    self.chess_ai.learn_from_game(final_reward, moves_made)
                    completed_games += 1
                    progress = (completed_games / num_games) * 100
                    self.root.after(0, self.update_training_progress, completed_games, progress)
                    self.root.after(0, self.update_metrics_display)
        
        self.root.after(0, self.finish_training)

    def update_training_progress(self, games_completed, progress):
        self.progress_bar['value'] = games_completed
        self.progress_var.set(f"Training progress: {progress:.1f}%")

    def finish_training(self):
        self.training_in_progress = False
        self.train_button.config(state='normal')
        self.progress_var.set("Training complete!")
        self.chess_ai.save_training_state(force=True)
        self.update_metrics_display()
        messagebox.showinfo("Training Complete", "Self-play training has finished!")

    def update_board(self):
        svg_data = chess.svg.board(self.board, size=self.canvas_size)
        png_data = cairosvg.svg2png(bytestring=svg_data.encode())
        image = Image.open(io.BytesIO(png_data))
        self.photo = ImageTk.PhotoImage(image)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        status = f"{turn} to move"
        if self.board.is_checkmate():
            status = f"Checkmate! {'Black' if self.board.turn == chess.WHITE else 'White'} wins!"
        elif self.board.is_stalemate():
            status = "Stalemate!"
        elif self.board.is_check():
            status = f"{turn} is in check!"
        self.status_var.set(status)

    def on_square_click(self, event):
        if self.training_in_progress:
            return
            
        file = event.x // self.square_size
        rank = 7 - (event.y // self.square_size)
        square = chess.square(file, rank)

        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.highlight_legal_moves(square)
        else:
            move = chess.Move(self.selected_square, square)
            
            if (move in self.board.legal_moves and 
                self.board.piece_at(self.selected_square) and 
                self.board.piece_at(self.selected_square).piece_type == chess.PAWN):
                if rank == 7 or rank == 0:
                    move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
            
            if move in self.board.legal_moves:
                self.board.push(move)
                self.update_board()
                if self.has_ai and not self.board.is_game_over():
                    self.root.after(500, self.make_ai_move)
            
            self.selected_square = None
            self.canvas.delete("highlight")

    def highlight_legal_moves(self, square):
        self.canvas.delete("highlight")
        for move in self.board.legal_moves:
            if move.from_square == square:
                x = (move.to_square & 7) * self.square_size
                y = (7 - (move.to_square >> 3)) * self.square_size
                self.canvas.create_oval(x+self.square_size/3, y+self.square_size/3, 
                                     x+self.square_size*2/3, y+self.square_size*2/3,
                                     fill="yellow", tags="highlight")

    def make_ai_move(self):
        if self.has_ai and not self.board.is_game_over():
            try:
                move = self.chess_ai.get_best_move(self.board)
                if move:
                    self.chess_ai.store_move(self.board)
                    self.board.push(move)
                    
                    if self.board.is_game_over():
                        final_reward = 1 if self.board.is_checkmate() else 0
                        self.chess_ai.learn_from_game(final_reward, len(self.board.move_stack))
                        
                    self.update_board()
                    self.update_metrics_display()
            except Exception as e:
                logging.error(f"AI error: {e}")
                messagebox.showerror("Error", f"AI error: {str(e)}")

    def new_game(self):
        if self.training_in_progress:
            return
            
        if self.has_ai and self.board.move_stack:
            if not self.board.is_game_over():
                self.chess_ai.save_training_state(force=True)
        self.board = chess.Board()
        self.selected_square = None
        self.update_board()

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    multiprocessing.set_start_method('spawn')
    root = tk.Tk()
    app = ChessGUI(root)
    root.mainloop()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()