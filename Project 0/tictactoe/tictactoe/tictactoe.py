"""
Tic Tac Toe Player
"""
import copy
import math

X = "X"
O = "O"
EMPTY = None

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]



def player(board):

    # count the number of X and O on the board
    countX = board[0].count(X)+board[1].count(X)+board[2].count(X)
    countO = board[0].count(O)+board[1].count(O)+board[2].count(O)
    if (countX - countO) == 0:
        return X
    else:
        return O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == EMPTY:
                actions.add((i,j))
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    b = copy.deepcopy(board)
    if player(board) == X:
        b[action[0]][action[1]] = X
    else:
        b[action[0]][action[1]] = O
    return b


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    if not terminal(board):
        return None
    if board[0][0] == board[0][1] == board[0][2] == X:
        return X
    elif board[0][0] == board[1][0] == board[2][0] == X:
        return X
    elif board[0][0] == board[1][1] == board[2][2] == X:
        return X
    elif board[0][1] == board[1][1] == board[2][1] == X:
        return X
    elif board[1][0] == board[1][1] == board[1][2] == X:
        return X
    elif board[2][0] == board[2][1] == board[2][2] == X:
        return X
    elif board[2][2] == board[1][2] == board[0][2] == X:
        return X
    elif board[2][0] == board[1][1] == board[0][2] == X:
        return X
    elif board[0][0] == board[0][1] == board[0][2] == O:
        return O
    elif board[0][0] == board[1][0] == board[2][0] == O:
        return O
    elif board[0][0] == board[1][1] == board[2][2] == O:
        return O
    elif board[0][1] == board[1][1] == board[2][1] == O:
        return O
    elif board[1][0] == board[1][1] == board[1][2] == O:
        return O
    elif board[2][0] == board[2][1] == board[2][2] == O:
        return O
    elif board[2][2] == board[1][2] == board[0][2] == O:
        return O
    elif board[2][0] == board[1][1] == board[0][2] == O:
        return O
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Check if all cells are filled
    # Check if someone has won
    if board[0].count(None) == board[1].count(None) == board[2].count(None) == 0:
        return True
    else:
        if board[0][0] == board[0][1] == board[0][2] != EMPTY:
            return True
        elif board[0][0] == board[1][0] == board[2][0] != EMPTY:
            return True
        elif board[0][0] == board[1][1] == board[2][2] != EMPTY:
            return True
        elif board[0][1] == board[1][1] == board[2][1] != EMPTY:
            return True
        elif board[1][0] == board[1][1] == board[1][2] != EMPTY:
            return True
        elif board[2][0] == board[2][1] == board[2][2] != EMPTY:
            return True
        elif board[2][2] == board[1][2] == board[0][2] != EMPTY:
            return True
        elif board[2][0] == board[1][1] == board[0][2] != EMPTY:
            return True
        else:
            return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    won = winner(board)
    if won == X:
        return 1
    elif won == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    value,move = miniMaxUtility(board,player(board)==X)
    return move


def miniMaxUtility(board, isMaxPlayer):
    # is the game done? if it is return board
    # get all possible actions
    if terminal(board):
        return utility(board),None

    moves = actions(board)
    om = None
    if(isMaxPlayer):
        v = -100
        for action in moves:
            temp, a = miniMaxUtility(result(board,action),False)
            if temp >v :
                v = temp
                om = action
            if v == 1:
                return v,om
        return v,om

    else:
        v = 100
        for action in moves:
            temp, a = miniMaxUtility(result(board,action),True)
            if temp < v:
                v = temp
                om = action
            if v == -1:
                return v,om
        return v,om

