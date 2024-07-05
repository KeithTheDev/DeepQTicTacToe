import random
import sys

import pygame
from tic_tac_toe.environment import TicTacToe
from tic_tac_toe.train import choose_action


def run_pygame(policy_net, target_net):
    pygame.init()

    WIDTH, HEIGHT = 300, 300
    LINE_WIDTH = 5
    BOARD_ROWS, BOARD_COLS = 3, 3
    SQUARE_SIZE = WIDTH // BOARD_COLS
    CIRCLE_RADIUS = SQUARE_SIZE // 3
    CIRCLE_WIDTH = 15
    CROSS_WIDTH = 25
    SPACE = SQUARE_SIZE // 4

    BG_COLOR = (28, 170, 156)
    LINE_COLOR = (23, 145, 135)
    CIRCLE_COLOR = (239, 231, 200)
    CROSS_COLOR = (84, 84, 84)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Tic Tac Toe')

    font = pygame.font.Font(None, 100)

    env = TicTacToe()

    ai_wins = 0
    human_wins = 0
    ties = 0

    def draw_lines():
        screen.fill(BG_COLOR)
        for row in range(1, BOARD_ROWS):
            pygame.draw.line(screen, LINE_COLOR, (0, row * SQUARE_SIZE), (WIDTH, row * SQUARE_SIZE), LINE_WIDTH)
            pygame.draw.line(screen, LINE_COLOR, (row * SQUARE_SIZE, 0), (row * SQUARE_SIZE, HEIGHT), LINE_WIDTH)

    def draw_figures():
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if env.board[row][col] == 1:
                    pygame.draw.line(screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE), 
                                     (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), CROSS_WIDTH)
                    pygame.draw.line(screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), 
                                     (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE), CROSS_WIDTH)
                elif env.board[row][col] == -1:
                    pygame.draw.circle(screen, CIRCLE_COLOR, 
                                       (int(col * SQUARE_SIZE + SQUARE_SIZE // 2), int(row * SQUARE_SIZE + SQUARE_SIZE // 2)), CIRCLE_RADIUS, CIRCLE_WIDTH)

    def check_winner(player):
        for row in range(BOARD_ROWS):
            if all([env.board[row][col] == player for col in range(BOARD_COLS)]):
                return True
        for col in range(BOARD_COLS):
            if all([env.board[row][col] == player for row in range(BOARD_ROWS)]):
                return True
        if all([env.board[i][i] == player for i in range(BOARD_ROWS)]) or all([env.board[i][BOARD_ROWS - i - 1] == player for i in range(BOARD_ROWS)]):
            return True
        return False

    def draw_winner(player):
        nonlocal ai_wins, human_wins, ties
        if player == 1:
            text = 'X wins!'
            human_wins += 1
        elif player == -1:
            text = 'O wins!'
            ai_wins += 1
        else:
            text = 'It\'s a tie!'
            ties += 1
        label = font.render(text, True, CIRCLE_COLOR if player == -1 else CROSS_COLOR)
        screen.blit(label, (WIDTH // 2 - label.get_width() // 2, HEIGHT // 2 - label.get_height() // 2))
        pygame.display.update()
        pygame.time.delay(2000)

    def draw_score():
        score_text = f"AI: {ai_wins} Human: {human_wins} Ties: {ties}"
        label = font.render(score_text, True, (255, 255, 255))
        screen.blit(label, (10, 10))

    def reset_game():
        draw_score()
        env.reset()
        draw_lines()
        draw_figures()
        new_player = -1 if random.random() < 0.5 else 1
        print(f"New game starting. Player {'O' if new_player == -1 else 'X'} goes first.")
        return new_player

    draw_lines()
    player = 1
    game_over = False

    while True:
        if player == -1 and not game_over:
            if env.available_actions():
                state = env.board
                action = choose_action(state, policy_net, 0, env)
                env.step(action, player)
                if check_winner(player):
                    game_over = True
                    draw_winner(player)
                    player = reset_game()
                    game_over = False
                else:
                    player *= -1
                draw_figures()
            else:
                game_over = True
                draw_winner(0)
                player = reset_game()
                game_over = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                mouseX = event.pos[0]
                mouseY = event.pos[1]
                clicked_row = mouseY // SQUARE_SIZE
                clicked_col = mouseX // SQUARE_SIZE

                if env.board[clicked_row][clicked_col] == 0:
                    env.board[clicked_row][clicked_col] = player
                    if check_winner(player):
                        game_over = True
                        draw_winner(player)
                        player = reset_game()
                        game_over = False
                    else:
                        player *= -1
                    draw_figures()

        pygame.display.update()
