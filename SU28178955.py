
import stdio
import sys

ERRORS = {
    "illegal" : "ERROR: Illegal argument",
    "few_args" : "ERROR: Too few arguments",
    "many_args" : "ERROR: Too many arguments",
    "sink_wrong_pos" : "ERROR: Sink in the wrong position",
    "piece_wrong_pos" : "ERROR: Piece in the wrong position",
    "piece_not_owned" : "ERROR: Piece does not belong to the correct player",
    "d_second_turn" : "ERROR: Cannot move a 2x2x2 piece on the second move",
    "beyond_board" : "ERROR: Cannot move beyond the board",

    "not_on_board(r,c)" : lambda r,c : f"ERROR: Field {r} {c} not on board",
    "invalid_object(o)" : lambda o: f"ERROR: Invalid object type {o}",
    "invalid_piece(p)" : lambda p: f"ERROR: Invalid piece type {p}",
    "invalid_direction(d)" : lambda d : f"ERROR: Invalid direction {d}",
    "no_piece(r,c)" : lambda r,c : f"ERROR: No piece on field {r} {c}",
    "field_not_free(r,c)" : lambda r,c : f"ERROR: Field {r} {c} not free"
    
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
        return not (board[max_row-row-1][col] == piece_code or board[max_row-row][col+1] == piece_code) # check if up or right has piece code

def print_board( board ):

    height, width = (len(board), len(board[0]))
    stdio.writeln(f"   {'  '.join([str(x) for x in range(width)])}  ")

    for rdx in range(height*2+1):

        space, line, board_row = (' ', '', round((rdx-1)/2.0))

        if rdx % 2 == 0:
            line += f'  {"".join(["+--" for x in range(width)])}+'
        else:
            line += f'{round(height-1-board_row)} |{"".join([f"{ space*(2-len(str(board[board_row][x])))+str(board[board_row][x]) }|" for x in range(width)])}'

        stdio.writeln(line)

def read_stdin_setup_to_board( board ):

    valid_setup = True

    while True:

        # Readline or exit if EOF (for testing etc.)
        try: line = stdio.readLine()
        except: break

        line_arguments = line.split(" ")


        if line_arguments[0] == "#":
            if valid_setup:
                print_board(board)
            break

        # Place sinks
        elif line_arguments[0] == "s":

            # Argument Length Check
            if len(line_arguments) < 4:
                stdio.writeln(ERRORS["few_args"])
                valid_setup = False
                continue
            elif len(line_arguments) > 4:
                stdio.writeln(ERRORS["many_args"])
                valid_setup = False
                continue

            # Unpack arguments
            _, size, row, col = line_arguments
            size, row, col = (int(size), int(row), int(col))

            # Piece check
            if size not in [1,2]:
                stdio.writeln(ERRORS["invalid_piece(p)"](size))
                valid_setup = False
                continue

            # Coordinate check
            if not check_coordinates_range_inclusive(row, col, 0, args["board_height"]-1, 0, args["board_width"]-1):
                stdio.writeln(ERRORS['not_on_board(r,c)'](row,col))
                valid_setup = False
                continue

            # Size check
            if size == 2:
                if not check_coordinates_range_inclusive(row, col, 0, args["board_height"]-2, 0, args["board_width"]-2):
                    stdio.writeln(ERRORS["sink_wrong_pos"])
                    valid_setup = False
                    continue

            # Location check
            if size == 1:
                if not( (row < 3 or row > args["board_height"]-4) or (col < 3 or col > args["board_width"]-4)):
                    stdio.writeln(ERRORS['sink_wrong_pos'])
                    valid_setup = False
                    continue
            else:
                if not( (row < 2 or row > args["board_height"]-4) or (col < 2 or col > args["board_width"]-4)):
                    stdio.writeln(ERRORS['sink_wrong_pos'])
                    valid_setup = False
                    continue

            # Adjacency check
            if not check_no_sink_adjacency(board, row, col, size):
                stdio.writeln(ERRORS['sink_wrong_pos'])
                valid_setup = False
                continue

            for r in range(size):
                for c in range(size):
                    board[args["board_height"]-row-r-1][col+c] = "s"

        # Place blocked tiles
        elif line_arguments[0] == "x":

            # Argument Length Check
            if len(line_arguments) < 3:
                stdio.writeln(ERRORS["few_args"])
                valid_setup = False
                continue
            elif len(line_arguments) > 3:
                stdio.writeln(ERRORS["many_args"])
                valid_setup = False
                continue

            # Unpack arguments
            _, row, col = line_arguments
            row, col = (int(row), int(col))

            if not check_coordinates_range_inclusive(row, col, 0, args["board_height"]-1, 0, args["board_width"]-1):
                stdio.writeln(ERRORS['not_on_board(r,c)'](row,col))
                valid_setup = False
                continue

            board[args["board_height"]-row-1][col] = "x"

        # Place Pieces
        elif line_arguments[0] == "d" or line_arguments[0] == "l":

            # Argument Length Check
            if len(line_arguments) < 4:
                stdio.writeln(ERRORS["few_args"])
                valid_setup = False
                continue
            elif len(line_arguments) > 4:
                stdio.writeln(ERRORS["many_args"])
                valid_setup = False
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
                valid_setup = False
                continue

            # position check
            if not check_coordinates_range_inclusive(row, col, 3, args["board_height"]-4, 3, args["board_width"]-4):
                stdio.writeln(ERRORS["piece_wrong_pos"])
                valid_setup = False
                continue
            
            # Free field check
            if board[args["board_height"]-row-1][col] != '':
                stdio.writeln(ERRORS["field_not_free(r,c)"](row,col))
                valid_setup = False
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
                valid_setup = False

        else:
            stdio.writeln(ERRORS["invalid_object(o)"](line_arguments[0]))
            valid_setup = False

def check_win_conditions( board ):
    return False

def move_pieces ( board, command, lights_turn, turn_number ):
    
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

        piece_upright = check_piece_upright(board, row, col) # True if piece occupies 1 slot
        light_team = (lambda x: True if x.lower() == x else False)(board[max_row-row][col]) # True if piece owned by light team
        piece_type_raw = board[max_row-row][col]
        piece_type = piece_type_raw.lower() # Get type to determine how movement will function
        piece_code = f"{row*(max_col+1) + col}"

        # Check if piece not owned
        if not (light_team == lights_turn): stdio.writeln(ERRORS["piece_not_owned"]); exit()

        # Moving type A blocks
        if piece_type == "a":
            
            # Only one type of movement
            directions = {'u' : (-1,0), 'd' : (1,0), 'l' : (0,-1), 'r' : (0,1)}
            direction = directions[move_type]

            # Check out of bounds
            if not check_coordinates_range_inclusive(max_row-row+direction[0],col+direction[1],0,max_row,0,max_col): stdio.writeln(ERRORS["beyond_board"]); exit()
            
            # Check for obstructions/sinks
            if board[max_row-row+direction[0]][col+direction[1]] == "s":
                board[max_row-row][col] = ""
            
            # Board space is occupied
            elif board[max_row-row+direction[0]][col+direction[1]] != '':
                stdio.writeln(ERRORS["field_not_free(r,c)"](max_row-row+direction[0],col+direction[1]))
                return
            
            # Move piece
            else:
                board[max_row-row+direction[0]][col+direction[1]] = piece_type_raw
                board[max_row-row][col] = ""

        # Moving B or C blocks
        elif piece_type in list("bc"):

            sizes = {'b' : 2, "c" : 3}
            directions = {'u' : (-1,0), 'd' : (1,0), 'l' : (0,-1), 'r' : (0,1)}

            direction = directions[move_type]
            size = sizes[piece_type]
            
            if piece_upright:

                if not check_coordinates_range_inclusive(max_row-row+direction[0]*size, col+direction[1]*size,0, max_row, 0, max_col): stdio.writeln(ERRORS["beyond_board"]); exit()

                # Check for no obstructions
                valid = True
                all_sinks = True
                obstructed = []

                for i in range(1,size+1):
                    if board[max_row-row+direction[0]*i][col+direction[1]*i]:
                        valid = False
                        obstructed.append(([row-direction[0]*i],[col+direction[1]*i]))
                    if not board[max_row-row+direction[0]*i][col+direction[1]*i] == "s":
                        all_sinks = False

                # Sink piece
                if all_sinks:
                    board[max_row-row][col] = ''
                    return

                obstructed.sort(key=lambda x : x[0]+x[1])
                if not valid: stdio.writeln(ERRORS["field_not_free(r,c)"](obstructed[0][0],obstructed[0][1])); exit()


                # Find new origin and calculate piece code
                affected_tiles = []
                for i in range(1,size+1):
                    affected_tiles.append((row-direction[0]*i,col+direction[1]*i))
                board[max_row-row][col] = ''

                affected_tiles.sort(key=lambda x : x[0]+x[1])
                origin_tile = affected_tiles[0]
                new_code = f"{origin_tile[0]*(max_col+1) + origin_tile[1]}"

                # Write new origin and coordinates                
                for tile in affected_tiles:
                    board[max_row-tile[0]][tile[1]] = new_code
                board[max_row-origin_tile[0]][origin_tile[1]] = piece_type_raw


            else:
                
                # Two types of movement. Lateral rolling or rolling upright
                directions = {"u" : (-1,0), "d" : (1,0), "l" : (0,-1), "r" : (0,1)}
                direction = directions[move_type]
                sizes = {"b" : 2, "c" : 3}
                size = sizes[piece_type]

                
                # Find piece alignment
                vertical = None
                if row == max_row:
                    vertical = False
                elif board[max_row-row-1][col] != piece_code:
                    vertical = False
                else:
                    vertical = True


                # Vertically aligned
                if vertical:
                    
                    # Roll over
                    if move_type in list("lr"):
                        
                        valid = True
                        all_sinks = True
                        obstructions = []

                        # Check coordinates
                        if not check_range_inclusive(col+direction[1],0,max_col): stdio.writeln(ERRORS["beyond_board"]); exit()

                        # Check destination
                        for i in range(size):
                            slot = board[max_row-row-i][col+direction[1]]
                            if slot != '': valid = False; obstructions.append((row+i,col+direction[1]))
                            elif slot != 's': all_sinks = False
                        
                        # Sink a piece
                        if all_sinks:
                            
                            # Clear piece and connected codes
                            for i in range(size):
                                board[max_row-row-i][col] = ""
                        
                        # Move piece
                        elif valid:
                            
                            # Copy origin
                            board[max_row-row][col+direction[1]] = board[max_row-row][col]
                            board[max_row-row][col] = ''
                            new_code = f"{row*(max_col+1) + col + direction[1]}"

                            # clear piece and connected codes and copy to new destinations
                            for i in range(1,size):
                                board[max_row-row-i][col] = ""
                                board[max_row-row-i][col+direction[1]] = new_code
                        
                        # Obstructions
                        else:
                            stdio.writeln(ERRORS["field_not_free(r,c)"](obstructions[0][0],obstructions[0][1]))
                    

                    # Flip upright
                    else:
                        destination = None
                        destination_coordinates = (0,0)
                        if move_type == 'u':
                            if not check_range_inclusive(max_row-row-size,0,max_row): stdio.writeln(ERRORS["beyond_board"]); exit()
                            destination = board[max_row-row-size][col]
                            destination_coordinates = (max_row-row-size,col)
                        else:
                            if not check_range_inclusive(max_row-row+1,0,max_row): stdio.writeln(ERRORS["beyond_board"]); exit()
                            destination = board[max_row-row+1][col]
                            destination_coordinates = (max_row-row+1,col)

                        if destination == "s":
                            for i in range(size):
                                board[max_row-row-i][col] = ''
                        elif destination == '':
                            for i in range(size):
                                board[max_row-row-i][col] = ''
                            board[destination_coordinates[0]][destination_coordinates[1]] = piece_type_raw
                        else:
                            stdio.writeln(ERRORS["field_not_free(r,c)"](max_row-destination_coordinates[0],destination_coordinates[1]))
                
                # Horizontally aligned
                else:
                    pass
                    
                    # Flip upright

                    # Roll over
        
        # Moving a sink or a 2x2 block
        elif piece_type == "d":
            
            # Validate second turn d piece moving
            if turn_number == 2: stdio.writeln(ERRORS["d_second_turn"]); exit()


        elif piece_type == "s":
            pass

        # Valid piece not at coords given
        else:
            stdio.writeln(ERRORS["no_piece(r,c)"](row,col))
            exit()
        
    else:
        stdio.writeln(ERRORS["invalid_direction(d)"](move_type))
        exit()



def main( args ):

    # Board setup
    board = [["" for x in range(args["board_width"])] for y in range(args["board_height"])]
    read_stdin_setup_to_board(board)

    # Gameloop var
    lights_turn = True
    turn_counter = 0

    # Gameloop
    while True:
        
        # Keep track of turns
        turn_counter += 1
        if turn_counter == 3: turn_counter = 1; lights_turn = not lights_turn

        # Read command or end partial game
        try: command = stdio.readLine()
        except: break

        move_pieces(board, command, lights_turn, turn_counter)
        print_board(board)
        
        # Check for win conditions
        if check_win_conditions(board):
            break
        



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
