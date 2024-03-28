
import stdio
import sys

ERRORS = {
    "illegal" : "ERROR: Illegal argument",
    "few_args" : "ERROR: Too few arguments",
    "many_args" : "ERROR: Too many arguments",
    "sink_wrong_pos" : "ERROR: Sink in the wrong position",
    "piece_wrong_pos" : "ERROR: Piece in the wrong position",

    "not_on_board(r,c)" : lambda r,c : f"ERROR: Field {r} {c} not on board",
    "invalid_object(o)" : lambda o: f"ERROR: Invalid object type {o}",
    "invalid_piece(p)" : lambda p: f"ERROR: Invalid piece type {p}",
    
}

# Validators
def check_range_inclusive(x , lower, upper):
    if x >= lower and x <= upper:
        return True
    return False

def check_coordinates_range_inclusive(y,x,y1,y2,x1,x2):
    if check_range_inclusive(y,y1,y2) and check_range_inclusive(x,x1,x2):
        return True
    return False

def check_no_sink_adjacency( board, row, col, sink_type ):
    
    def check_tile(board, row, col):

        max_row, max_col = (len(board)-1, len(board[0])-1)
        internal_row = max_row - row
        
        up = internal_row - 1
        down = internal_row + 1
        left = col - 1
        right = col + 1

        valid = True

        if check_range_inclusive(up, 0, max_row):
            if board[up][col] == "s":
                valid = False

        if check_range_inclusive(down, 0, max_row):
            if board[down][col] == "s":
                valid = False

        if check_range_inclusive(left, 0, max_col):
            if board[internal_row][left] == "s":
                valid = False

        if check_range_inclusive(right, 0, max_col):
            if board[internal_row][right] == "s":
                valid = False

        return valid

    if sink_type == 1:
        return check_tile(board,row,col)
    else:
        valid = True
        for y in range(2):
            for x in range(2):
                if not check_tile(board, row+y, col+x):
                    valid = False

    return valid

def check_piece_upright( board, row, col ):
    
    max_row, max_col = (len(board)-1, len(board[0])-1)
    piece_type = board[max_row-row][col]
    piece_code = f"{row*(max_col+1) + col}"

    if piece_type in list("dD"):
        return False
    else:

        # 4 Cases to check
        if row == max_row:
            if col == max_col:
                return True # BL coord is TR must be upright
            return not board[max_row-row][col+1] == piece_code # check if right has piece code

        if col == max_col:
            return not board[max_row-row-1][col] == piece_code # check if up has piece code
        return not (board[max_row-row-1][col] == piece_code or board[max_row-row][col+1] == piece_code) # check if up or left has piece code

def print_board( board ):

    height, width = (len(board), len(board[0]))
    stdio.writeln(f"   {'  '.join([str(x) for x in range(width)])}  ")

    for rdx in range(height*2+1):

        space, empty, line, board_row = (' ', '', '', round((rdx-1)/2.0))

        if rdx % 2 == 0:
            line += f'  {"".join(["+--" for x in range(width)])}+'
        else:
            line += f'{round(height-1-board_row)} |{"".join([f"{ (lambda char : space if len(str(char)) < 2 else empty)(board[board_row][x]) }{board[board_row][x]}|" for x in range(width)])}'

        stdio.writeln(line)

def read_stdin_setup_to_board( board ):

    while True:

        # Readline or exit if EOF (for testing etc.)
        try: line = stdio.readLine()
        except: break

        line_arguments = line.split(" ")


        if line_arguments[0] == "#":
            print_board(board)
            break

        # Place sinks
        elif line_arguments[0] == "s":

            # Argument Length Check
            if len(line_arguments) < 4:
                stdio.writeln(ERRORS["few_args"])
                continue
            elif len(line_arguments) > 4:
                stdio.writeln(ERRORS["many_args"])
                continue

            # Unpack arguments
            _, size, row, col = line_arguments
            size, row, col = (int(size), int(row), int(col))

            # Piece check
            if size not in [1,2]:
                stdio.writeln(ERRORS["invalid_piece(p)"](size))
                continue

            # Coordinate check
            if not check_coordinates_range_inclusive(row, col, 0, args["board_height"]-1, 0, args["board_width"]-1):
                stdio.writeln(ERRORS['not_on_board(r,c)'](row,col))
                continue

            # Size check
            if size == 2:
                if not check_coordinates_range_inclusive(row, col, 0, args["board_height"]-2, 0, args["board_width"]-2):
                    stdio.writeln(ERRORS["sink_wrong_pos"])
                    continue

            # Location check
            if size == 1:
                if not( (row < 3 or row > args["board_height"]-4) or (col < 3 or col > args["board_width"]-4)):
                    stdio.writeln(ERRORS['sink_wrong_pos'])
                    continue
            else:
                if not( (row < 2 or row > args["board_height"]-4) or (col < 2 or col > args["board_width"]-4)):
                    stdio.writeln(ERRORS['sink_wrong_pos'])
                    continue

            # Adjacency check
            if not check_no_sink_adjacency(board, row, col, size):
                stdio.writeln(ERRORS['sink_wrong_pos'])
                continue

            for r in range(size):
                for c in range(size):
                    board[args["board_height"]-row-r-1][col+c] = "s"

        # Place blocked tiles
        elif line_arguments[0] == "x":

            # Argument Length Check
            if len(line_arguments) < 3:
                stdio.writeln(ERRORS["few_args"])
                continue
            elif len(line_arguments) > 3:
                stdio.writeln(ERRORS["many_args"])
                continue

            # Unpack arguments
            _, row, col = line_arguments
            row, col = (int(row), int(col))

            if not check_coordinates_range_inclusive(row, col, 0, args["board_height"]-1, 0, args["board_width"]-1):
                stdio.writeln(ERRORS['not_on_board(r,c)'](row,col))
                continue

            board[args["board_height"]-row-1][col] = "x"

        # Place Pieces
        elif line_arguments[0] == "d" or line_arguments[0] == "l":

            # Argument Length Check
            if len(line_arguments) < 4:
                stdio.writeln(ERRORS["few_args"])
                continue
            elif len(line_arguments) > 4:
                stdio.writeln(ERRORS["many_args"])
                continue
            
            # Unpack arguments
            team, piece_type, row, col = line_arguments
            row, col = (int(row), int(col))

            # Choose board id
            if team == "d": id = piece_type.upper()
            else: id = piece_type

            # Check coords
            if not check_coordinates_range_inclusive(row, col, 0, args["board_height"]-1, 0, args["board_width"]-1):
                stdio.writeln(ERRORS['not_on_board(r,c)'](row,col))
                continue

            # position check
            if not check_coordinates_range_inclusive(row, col, 3, args["board_height"]-4, 3, args["board_width"]-4):
                stdio.writeln(ERRORS["piece_wrong_pos"])
                continue

            # Place pieces    
            if piece_type in ['a','b','c']:
                board[args["board_height"]-row-1][col] = id

            elif piece_type == 'd':

                root_id = int(row)*args["board_width"]+int(col)
                board[args["board_height"]-row-1][col] = id
                board[args["board_height"]-row-2][col] = root_id
                board[args["board_height"]-row-2][col+1] = root_id
                board[args["board_height"]-row-1][col+1] = root_id
                
            else:
                stdio.writeln(ERRORS["invalid_piece(p)"](piece_type))

        else:
            stdio.writeln(ERRORS["invalid_object(o)"](line_arguments[0]))


def check_win_conditions( board ):
    return False

def move_pieces ( board, command, lights_turn ):
    
    args = command.split(" ")
    max_row, max_col = (len(board)-1, len(board[0])-1)

    # Validate arguments
    if len(args) < 3:
        stdio.writeln(ERRORS["few_args"])
        return 
    elif len(args) > 3:
        stdio.writeln(ERRORS["many_args"])
        return
    
    # Datatype and range checks
    try: row, col, move_type = (int(args[0]), int(args[1]), args[2])
    except: stdio.writeln(ERRORS["illegal"]); return
    if not check_coordinates_range_inclusive(row, col, 0, max_row, 0, max_col): stdio.writeln(ERRORS["not_on_board(r,c)"](row,col)); return


    # Moving pieces
    if move_type in list("udlr"):

        piece_upright = check_piece_upright(board, row, col)
        light_team = (lambda x: True if x.lower() == x else False)(board[max_row-row][col])
        piece_type = board[max_row-row][col].lower()

        # Check if piece not owned
        if not (light_team == lights_turn): stdio.writeln(ERRORS["illegal"]); return

        # Moving A, B or C blocks
        if piece_type in list("abc"):

            sizes = {'a' : 1, 'b' : 2, "c" : 3}
            size = sizes[piece_type]

            # Range check
            if not (row <= max_row-size):
                stdio.writeln(ERRORS["illegal"])
                return
                
            if move_type == "u":
                pass
            elif move_type =="d":
                pass
            if move_type == "l":
                pass
            elif move_type =="r":
                pass
        
        # Moving a sink or a 2x2 block
        elif piece_type in list("ds"):
            pass

        # Valid piece not at coords given
        else:
            stdio.writeln(ERRORS["illegal"])
            return



def main( args ):

    # Board setup
    board = [[" " for x in range(args["board_width"])] for y in range(args["board_height"])]
    read_stdin_setup_to_board(board)

    # Gameloop var
    lights_turn = True
    turn = 1

    # Gameloop
    while True:

        # Read command or end partial game
        try: command = stdio.readLine()
        except: break

        move_pieces(board, command, lights_turn)
        print_board(board)
        
        # Check for win conditions
        if check_win_conditions(board):
            break

        # increment loopvar
        lights_turn = not lights_turn
        turn += 1



# Program Entry point
if __name__ == "__main__":
    
    # CL Input
    valid_args = True
    args = sys.argv


    # Too Few arguments
    if len(args)-1 < 3:
        stdio.writeln(ERRORS['few_args'])

    # Too many arguments
    elif len(args)-1 > 3:
        stdio.writeln(ERRORS['many_args'])

    # Valid Argument Count
    else:

        # Data type check
        try:
            args = {"board_height" : int(args[1]), "board_width" : int(args[2]), "gui" : int(args[3])}

            # Range Check
            if not ( check_range_inclusive(args["board_height"], 8, 10) and check_range_inclusive(args["board_width"], 8, 10) and args["gui"] in [0,1]):
                stdio.writeln(ERRORS['illegal'])
                valid_args = False

        except:
            stdio.writeln(ERRORS['illegal'])
            valid_args = False


        # Check validation and start
        if valid_args:
            main(args)
