#!/usr/bin/env python3
from FourConnect import *  # See the FourConnect.py file
import csv
import copy
from math import exp
import time

# INDEX:
# To change depth: Line 68
# Without move ordering code section: From Line 23, comment Lines 64-109
# Evaluation function 1: Line 112
# Evaluation function 2: Line 147
# Evaluation function 3: Line 210
# Evaluation function 4: Line 273. Note that only one eval func will work at a time.
# To change number of games for benchmarking or performance: Lines 440 and 475
# Testcase or play game: Line 511 and 512

class GameTreePlayer:
    def __init__(self):
        self.games_won = 0
        self.moves_per_game = []

# WITHOUT MOVE ORDERING HEURISTICS

    def FindBestAction(self, currentState):
        valid_actions = self.get_valid_actions(currentState)
        bestAction = self.minimax(currentState, depth=5, maximizingPlayer=True, alpha=float('-inf'), beta=float('inf'), valid_actions=valid_actions)[1]
        return bestAction

    def minimax(self, state, depth, maximizingPlayer, alpha, beta, valid_actions):
        if depth == 0 or self.game_over(state):  # Call game_over on the FourConnect instance
            return self.evaluate(state), None

        best_action = None

        if maximizingPlayer:
            max_eval = float('-inf')
            for action in valid_actions:
                child_state = self.get_next_state(state, action, player=2)
                eval, _ = self.minimax(child_state, depth - 1, False, alpha, beta, valid_actions)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_action

        else:
            min_eval = float('inf')
            for action in valid_actions:
                child_state = self.get_next_state(state, action, player=1)
                eval, _ = self.minimax(child_state, depth - 1, True, alpha, beta, valid_actions)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_action

# WITH MOVE ORDERING HEURISTICS: CENTER PRIORITY. Time difference in report.

    # def FindBestAction(self, currentState):
    #     # Changing our depth here (3 or 5)
    #     valid_actions = self.get_valid_actions(currentState)
    #     sorted_actions = self.order_actions_by_center_priority(valid_actions)
    #     bestAction = self.minimax(currentState, depth=5, maximizingPlayer=True, alpha=float('-inf'), beta=float('inf'), sorted_actions=sorted_actions)[1]
    #     return bestAction

    # def minimax(self, state, depth, maximizingPlayer, alpha, beta, sorted_actions):
    #     if depth == 0 or self.game_over(state):  # Call game_over on the FourConnect instance
    #         return self.evaluate(state), None

    #     valid_actions = sorted_actions  # Use sorted actions

    #     best_action = None

    #     if maximizingPlayer:
    #         max_eval = float('-inf')
    #         for action in valid_actions:
    #             child_state = self.get_next_state(state, action, player=2)
    #             eval, _ = self.minimax(child_state, depth - 1, False, alpha, beta, sorted_actions)
    #             if eval > max_eval:
    #                 max_eval = eval
    #                 best_action = action
    #             alpha = max(alpha, eval)
    #             if beta <= alpha:
    #                 break
    #         return max_eval, best_action

    #     else:
    #         min_eval = float('inf')
    #         for action in valid_actions:
    #             child_state = self.get_next_state(state, action, player=1)
    #             eval, _ = self.minimax(child_state, depth - 1, True, alpha, beta, sorted_actions)
    #             if eval < min_eval:
    #                 min_eval = eval
    #                 best_action = action
    #             beta = min(beta, eval)
    #             if beta <= alpha:
    #                 break
    #         return min_eval, best_action

    # def order_actions_by_center_priority(self, valid_actions):
    #     # Order actions based on center priority
    #     center_column = 3  # Center column index
    #     sorted_actions = sorted(valid_actions, key=lambda col: abs(col - center_column))
    #     return sorted_actions

# Evaluation Functions.
# 1: Corresponds to (i) in report

    # def evaluate(self, state):
    #     # Simple evaluation function: count player pieces in winning configurations
    #     player = 2  # Game Tree player is always Player 2
    #     opponent = 1  # Myopic player is always Player 1

    #     player_score = self.count_player_pieces(state, player)
    #     opponent_score = self.count_player_pieces(state, opponent)

    #     return player_score - opponent_score

    # def count_player_pieces(self, state, player):
    #     # Count player pieces in rows, columns, and diagonals
    #     score = 0

    #     # Rows
    #     for row in state:
    #         score += row.count(player)

    #     # Columns
    #     for col in range(7):
    #         score += sum(state[row][col] == player for row in range(6))

    #     # Diagonals
    #     for row in range(3):
    #         for col in range(4):
    #             score += sum(state[row + i][col + i] == player for i in range(4))

    #     for row in range(3, 6):
    #         for col in range(4):
    #             score += sum(state[row - i][col + i] == player for i in range(4))

    #     return score

# 2: Corresponds to (ii) in report

    # def evaluate(self, state):
    #     player = 2  # Game Tree player is always Player 2
    #     opponent = 1  # Myopic player is always Player 1

    #     player_score = self.count_player_pieces(state, player)
    #     opponent_score = self.count_player_pieces(state, opponent)

    #     return player_score - opponent_score

    # def count_player_pieces(self, state, player):
    #     score = 0

    #     # Rows
    #     for row in state:
    #         score += self.calculate_weight(row, player)

    #     # Columns
    #     for col in range(7):
    #         column = [state[row][col] for row in range(6)]
    #         score += self.calculate_weight(column, player)

    #     # Diagonals
    #     for row in range(3):
    #         for col in range(4):
    #             diagonal = [state[row + i][col + i] for i in range(4)]
    #             score += self.calculate_weight(diagonal, player)

    #     for row in range(3, 6):
    #         for col in range(4):
    #             diagonal = [state[row - i][col + i] for i in range(4)]
    #             score += self.calculate_weight(diagonal, player)

    #     return score

    # def calculate_weight(self, line, player):
    #     weight = 0
    #     count = 0

    #     for piece in line:
    #         if piece == player:
    #             count += 1
    #         else:
    #             weight += self.calculate_score(count)
    #             count = 0

    #     weight += self.calculate_score(count)

    #     return weight

    # def calculate_score(self, count):
    #     if count >= 4:
    #         return 4
    #     elif count == 3:
    #         return 3
    #     elif count == 2:
    #         return 2
    #     elif count == 1:
    #         return 1
    #     else:
    #         return 0

# 3: Corresponds to (iii) in report

    def evaluate(self, state):
        player = 2  # Game Tree player is always Player 2
        opponent = 1  # Myopic player is always Player 1

        player_score = self.count_player_pieces(state, player)
        opponent_score = self.count_player_pieces(state, opponent)

        return player_score - opponent_score

    def count_player_pieces(self, state, player):
        score = 0

        # Rows
        for row in state:
            score += self.calculate_weight(row, player)

        # Columns
        for col in range(7):
            column = [state[row][col] for row in range(6)]
            score += self.calculate_weight(column, player)

        # Diagonals
        for row in range(3):
            for col in range(4):
                diagonal = [state[row + i][col + i] for i in range(4)]
                score += self.calculate_weight(diagonal, player)

        for row in range(3, 6):
            for col in range(4):
                diagonal = [state[row - i][col + i] for i in range(4)]
                score += self.calculate_weight(diagonal, player)

        return score

    def calculate_weight(self, line, player):
        weight = 0
        count = 0

        for piece in line:
            if piece == player:
                count += 1
            else:
                weight += self.calculate_score(count)
                count = 0

        weight += self.calculate_score(count)

        return weight

    def calculate_score(self, count):
        if count >= 4:
            return 100000  # Very high weight for winning condition
        elif count == 3:
            return 1000
        elif count == 2:
            return 100
        elif count == 1:
            return 10
        else:
            return 0

# 4: Tried an additional evaluation function which proiritises center columns and gives more weight to
#    as many two-coin configurations for player as possible. However, for our purposes, the third one 
#    is most ideal. 

    # def evaluate(self, state):
    #     player = 2  # Game Tree player is always Player 2
    #     opponent = 1  # Myopic player is always Player 1

    #     player_score = self.count_player_pieces(state, player)
    #     opponent_score = self.count_player_pieces(state, opponent)

    #     return player_score - opponent_score + self.evaluate_board(state, player)

    # def count_player_pieces(self, state, player):
    #     score = 0

    #     # Rows
    #     for row in state:
    #         score += self.calculate_weight(row, player)

    #     # Columns
    #     for col in range(7):
    #         column = [state[row][col] for row in range(6)]
    #         score += self.calculate_weight(column, player)

    #     # Diagonals
    #     for row in range(3):
    #         for col in range(4):
    #             diagonal = [state[row + i][col + i] for i in range(4)]
    #             score += self.calculate_weight(diagonal, player)

    #     for row in range(3, 6):
    #         for col in range(4):
    #             diagonal = [state[row - i][col + i] for i in range(4)]
    #             score += self.calculate_weight(diagonal, player)

    #     return score
    
    # def calculate_weight(self, line, player):
    #     weight = 0
    #     count = 0

    #     for piece in line:
    #         if piece == player:
    #             count += 1
    #         else:
    #             weight += self.calculate_score(count)
    #             count = 0

    #     weight += self.calculate_score(count)

    #     return weight

    # def calculate_score(self, count):
    #     if count >= 4:
    #         return 100000  # Very high weight for winning condition
    #     elif count == 3:
    #         return 1000
    #     elif count == 2:
    #         return 100
    #     elif count == 1:
    #         return 10
    #     else:
    #         return 0

    # def evaluate_board(self, state, player):
    #     score = 0

    #     # Evaluate the center column
    #     center_column = [state[row][3] for row in range(6)]
    #     score += self.calculate_weight(center_column, player)

    #     # Evaluate the number of open two-in-a-row configurations
    #     score += 2 * self.count_open_two_in_a_row(state, player)

    #     return score

    # def count_open_two_in_a_row(self, state, player):
    #     count = 0

    #     for row in range(6):
    #         for col in range(7):
    #             if state[row][col] == 0:
    #                 # Check horizontally
    #                 if col + 1 < 7 and col - 1 >= 0 and state[row][col + 1] == player and state[row][col - 1] == player:
    #                     count += 1

    #                 # Check vertically
    #                 if row + 1 < 6 and row - 1 >= 0 and state[row + 1][col] == player and state[row - 1][col] == player:
    #                     count += 1

    #                 # Check diagonally (positive slope)
    #                 if row + 1 < 6 and col + 1 < 7 and row - 1 >= 0 and col - 1 >= 0 and \
    #                    state[row + 1][col + 1] == player and state[row - 1][col - 1] == player:
    #                     count += 1

    #                 # Check diagonally (negative slope)
    #                 if row - 1 >= 0 and col + 1 < 7 and row + 1 < 6 and col - 1 >= 0 and \
    #                    state[row - 1][col + 1] == player and state[row + 1][col - 1] == player:
    #                     count += 1

    #     return count



    def game_over(self, state):
        # Check if the board is full
        if all(state[row][col] != 0 for row in range(6) for col in range(7)):
            return True

        # Check for horizontal, vertical, and diagonal wins for both players
        for player in [1, 2]:
            for row in range(6):
                for col in range(7):
                    if state[row][col] == player:
                        # Check horizontally
                        if col + 3 < 7 and all(state[row][col + i] == player for i in range(4)):
                            return True

                        # Check vertically
                        if row + 3 < 6 and all(state[row + i][col] == player for i in range(4)):
                            return True

                        # Check diagonally (positive slope)
                        if row + 3 < 6 and col + 3 < 7 and all(state[row + i][col + i] == player for i in range(4)):
                            return True

                        # Check diagonally (negative slope)
                        if row - 3 >= 0 and col + 3 < 7 and all(state[row - i][col + i] == player for i in range(4)):
                            return True
        return False

    # Checking valid actions for game tree player
    def get_valid_actions(self, state):
        return [col for col in range(7) if state[0][col] == 0]

    # Getting next state after action of game tree player.
    def get_next_state(self, state, action, player):
        new_state = copy.deepcopy(state)
        row = 0
        while row < 6 and new_state[row][action] == 0:
            row += 1
        new_state[row - 1][action] = player
        return new_state


def LoadTestcaseStateFromCSVfile():
    testcaseState = list()

    with open('testcase.csv', 'r') as read_obj:
        csvReader = csv.reader(read_obj)
        for csvRow in csvReader:
            row = [int(r) for r in csvRow]
            testcaseState.append(row)
    return testcaseState


def PlayGame():
    # Initialize the algorithm
    gameTree = GameTreePlayer()

    # Initialize counters
    games_won = 0
    moves_per_game = []

    total_time_start = time.time()

    for _ in range(10):  # Play 50 games
        fourConnect = FourConnect()  # Create a new instance for each game
        move = 0

        game_start_time = time.time()

        while move < 42:
            if move % 2 == 0:
                fourConnect.MyopicPlayerAction()
            else:
                currentState = fourConnect.GetCurrentState()
                game_tree_start_time = time.time()
                gameTreeAction = gameTree.FindBestAction(currentState)
                game_tree_end_time = time.time()
                print(f"Game Tree Player took {game_tree_end_time - game_tree_start_time:.6f} seconds for move {move}")

                fourConnect.GameTreePlayerAction(gameTreeAction)

            move += 1

            if fourConnect.winner is not None:
                if fourConnect.winner == 2:
                    games_won += 1
                    moves_per_game.append(move)
                break

        game_end_time = time.time()
        print(f"Game took {game_end_time - game_start_time:.6f} seconds")

    total_time_end = time.time()
    total_time = total_time_end - total_time_start

    # Calculate average moves
    average_moves = sum(moves_per_game) / games_won if games_won > 0 else 0
    print(f"\nGames Won: {games_won}")
    print(f"Games Not Won: {10 - games_won}")
    print(f"Average Moves per Win: {average_moves}")
    print(f"Total time for all games: {total_time:.6f} seconds")


def RunTestCase():
    fourConnect = FourConnect()
    gameTree = GameTreePlayer()
    testcaseState = LoadTestcaseStateFromCSVfile()
    fourConnect.SetCurrentState(testcaseState)
    fourConnect.PrintGameState()

    move = 0
    while move < 5:  # Player 2 must win in 5 moves
        if move % 2 == 1:
            fourConnect.MyopicPlayerAction()
        else:
            currentState = fourConnect.GetCurrentState()
            gameTreeAction = gameTree.FindBestAction(currentState)
            fourConnect.GameTreePlayerAction(gameTreeAction)
        fourConnect.PrintGameState()
        move += 1
        if fourConnect.winner is not None:
            break

    print("Roll no : 2021A7PS2595G")  # Put your roll number here

    if fourConnect.winner == 2:
        print("Player 2 has won. Testcase passed.")
    else:
        print("Player 2 could not win in 5 moves. Testcase failed.")
    print("Moves : {0}".format(move))


def main():

    #RunTestCase()
    PlayGame()


if __name__ == '__main__':
    main()
