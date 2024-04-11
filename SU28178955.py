
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
    "repeated_position" : "ERROR: Piece cannot be returned to starting position",
    "no_sink_moves" : "ERROR: No sink moves left",
    "sink_adjacency" : "ERROR: Sink cannot be next to another sink",
    "frozen" : "ERROR: Cannot move frozen piece",   
    "no_freezes" : "ERROR: No freezings left",

    "not_on_board(r,c)" : lambda r,c : f"ERROR: Field {r} {c} not on board",
    "invalid_object(o)" : lambda o: f"ERROR: Invalid object type {o}",
    "invalid_piece(p)" : lambda p: f"ERROR: Invalid piece type {p}",
    "invalid_direction(d)" : lambda d : f"ERROR: Invalid direction {d}",
    "no_piece(r,c)" : lambda r,c : f"ERROR: No piece on field {r} {c}",
    "field_not_free(r,c)" : lambda r,c : f"ERROR: Field {r} {c} not free"
    
}

OUTCOMES = {
        "l_win" : "Light wins!",
        "l_lose" : "Light loses",
        "d_win" : "Dark wins!",
        "d_lose" : "Dark loses",
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
        except: exit()

        line_arguments = [x for x in line.split(" ") if x != '']
        if len(line_arguments) < 1:
            stdio.writeln(ERRORS["invalid_object(o)"]("''")); exit()


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
                exit()
            elif len(line_arguments) > 4:
                stdio.writeln(ERRORS["many_args"])
                valid_setup = False
                exit()

            # Unpack arguments
            _, size, row, col = line_arguments
            size, row, col = (int(size), int(row), int(col))

            # Piece check
            if size not in [1,2]:
                stdio.writeln(ERRORS["invalid_piece(p)"](size))
                valid_setup = False
                exit()

            # Coordinate check
            if not check_coordinates_range_inclusive(row, col, 0, args["board_height"]-1, 0, args["board_width"]-1):
                stdio.writeln(ERRORS['not_on_board(r,c)'](row,col))
                valid_setup = False
                exit()

            # Size check
            if size == 2:
                if not check_coordinates_range_inclusive(row, col, 0, args["board_height"]-2, 0, args["board_width"]-2):
                    stdio.writeln(ERRORS["sink_wrong_pos"])
                    valid_setup = False
                    exit()

            # Location check
            if size == 1:
                if not( (row < 3 or row > args["board_height"]-4) or (col < 3 or col > args["board_width"]-4)):
                    stdio.writeln(ERRORS['sink_wrong_pos'])
                    valid_setup = False
                    exit()
            else:
                if not( (row < 2 or row > args["board_height"]-4) or (col < 2 or col > args["board_width"]-4)):
                    stdio.writeln(ERRORS['sink_wrong_pos'])
                    valid_setup = False
                    exit()

            # Adjacency check
            if not check_no_sink_adjacency(board, row, col, size):
                stdio.writeln(ERRORS['sink_adjacency'])
                valid_setup = False
                exit()

            for r in range(size):
                for c in range(size):
                    board[args["board_height"]-row-r-1][col+c] = "s"

        # Place blocked tiles
        elif line_arguments[0] == "x":

            # Argument Length Check
            if len(line_arguments) < 3:
                stdio.writeln(ERRORS["few_args"])
                valid_setup = False
                exit()
            elif len(line_arguments) > 3:
                stdio.writeln(ERRORS["many_args"])
                valid_setup = False
                exit()

            # Unpack arguments
            _, row, col = line_arguments
            row, col = (int(row), int(col))

            if not check_coordinates_range_inclusive(row, col, 0, args["board_height"]-1, 0, args["board_width"]-1):
                stdio.writeln(ERRORS['not_on_board(r,c)'](row,col))
                valid_setup = False
                exit()

            board[args["board_height"]-row-1][col] = "x"

        # Place Pieces
        elif line_arguments[0] == "d" or line_arguments[0] == "l":

            # Argument Length Check
            if len(line_arguments) < 4:
                stdio.writeln(ERRORS["few_args"])
                valid_setup = False
                exit()
            elif len(line_arguments) > 4:
                stdio.writeln(ERRORS["many_args"])
                valid_setup = False
                exit()
            
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
                exit()

            # position check
            if not check_coordinates_range_inclusive(row, col, 3, args["board_height"]-4, 3, args["board_width"]-4):
                stdio.writeln(ERRORS["piece_wrong_pos"])
                valid_setup = False
                exit()
            
            # Free field check
            if board[args["board_height"]-row-1][col] != '':
                stdio.writeln(ERRORS["field_not_free(r,c)"](row,col))
                quit()

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
                quit()

        else:
            stdio.writeln(ERRORS["invalid_object(o)"](line_arguments[0]))
            quit()

def check_win_conditions( board, player_scores, freezes, sink_moves ):
    # Win via points
    if player_scores["l"] >= 4:
        stdio.writeln(OUTCOMES["l_win"])
        exit()
    elif player_scores["d"] >= 4:
        stdio.writeln(OUTCOMES["d_win"])
        exit()

def check_for_moves(board, lights_turn, turn_number, sink_moves, frozen_pieces, freezes):
    
    # init
    possible = 0
    owned_piece_locations = []
    possible_moves = list('udlr')
    max_row = len(board)-1

    # Find any sinks and add owned piece coordinates to a list for move bruteforcing
    sinks_present = False
    for rdx,row in enumerate(board):
        for cdx,col in enumerate(row):
            if col == "s": sinks_present = True
            
            char = str(board[rdx][cdx])
            if char.lower() in list("abcd"):
                capital = False if char.lower() == char else True

                if lights_turn != capital:
                    owned_piece_locations.append((rdx,cdx))

    # For each piece bruteforce possible moves
    for row, col in owned_piece_locations:
        for move in possible_moves:
            if move_pieces(board,f"{max_row-row} {col} {move}", lights_turn, turn_number, sink_moves, frozen_pieces, freezes, report_actions_left=True): possible += 1


    # Allow for possible sink shifting (CHECK IF THIS COUNTS AS A MOVE)
    if sinks_present: possible += sink_moves["l" if lights_turn else "d"]

    if possible == 0:
        if lights_turn: stdio.writeln(OUTCOMES["l_lose"]); quit()
        else: stdio.writeln(OUTCOMES["d_lose"]); quit()

def get_sink_size( board, row, col ):
    max_row = len(board)-1
    if row == max_row:
        return 1
    return 2 if board[max_row-row-1][col] == 's' else 1
    
def copy_board(board):
    return [row.copy() for row in board]
    
def move_pieces ( board, command, lights_turn, turn_number, sink_moves, frozen_pieces, freezes, report_actions_left=False ):
    
    args = [x for x in command.split(" ") if x != '']
    max_row, max_col = (len(board)-1, len(board[0])-1)
    digits = list("0123456789")

    # Validate arguments
    if len(args) < 3:
        if not report_actions_left : stdio.writeln(ERRORS["few_args"]); exit()
        else: return False 
    elif len(args) > 3:
        if not report_actions_left : stdio.writeln(ERRORS["many_args"]); exit()
        else: return False

    # Datatype and range checks
    try: row, col, move_type = (int(args[0]), int(args[1]), args[2])
    except: 
        if not report_actions_left : stdio.writeln(ERRORS["illegal"]); exit()
        else: return False

    if not check_coordinates_range_inclusive(row, col, 0, max_row, 0, max_col): 
        if not report_actions_left : stdio.writeln(ERRORS["not_on_board(r,c)"](row,col)); exit()
        else: return False

    if board[max_row-row][col] == '': 
        if not report_actions_left : stdio.writeln(ERRORS["no_piece(r,c)"](row,col)); exit()
        else: return False

    # If player refers to a coordinate
    if sum([ (lambda x : 0 if x in digits else 1 )(char) for char in str(board[max_row-row][col])]) == 0:
        code = int(board[max_row-row][col])
        col = code % (max_col+1)
        row = int((code - col)/(max_col+1))
    

    # Moving pieces
    # Validate direction
    if move_type in list("udlr"):

        piece_upright = check_piece_upright(board, row, col) # True if piece occupies 1 slot
        light_team = (lambda x: True if x.lower() == x else False)(board[max_row-row][col]) # True if piece owned by light team
        piece_type_raw = board[max_row-row][col]
        piece_type = piece_type_raw.lower() # Get type to determine how movement will function
        piece_code = f"{row*(max_col+1) + col}"

        # Check if piece not owned
        if not (light_team == lights_turn or piece_type == "s"):
            if not report_actions_left : stdio.writeln(ERRORS["piece_not_owned"]); exit()
            else: return False


        # Check if piece is frozen
        frozen = False
        for r, c, _, _ in frozen_pieces:
            if r == row and c == col:
                frozen = True

        if frozen: 
            if not report_actions_left : stdio.writeln(ERRORS['frozen']); exit()
            else: return False


        # Moving type A blocks
        if piece_type == "a":
            
            # Only one type of movement
            directions = {'u' : (-1,0), 'd' : (1,0), 'l' : (0,-1), 'r' : (0,1)}
            direction = directions[move_type]

            # Check out of bounds
            if not check_coordinates_range_inclusive(max_row-row+direction[0],col+direction[1],0,max_row,0,max_col): 
                if not report_actions_left : stdio.writeln(ERRORS["beyond_board"]); exit()
                else: return False
            
            # Check for obstructions/sinks
            if board[max_row-row+direction[0]][col+direction[1]] == "s":
                if report_actions_left:
                    return True
                else:
                    board[max_row-row][col] = ""
                    return ['s',1]
            
            # Board space is occupied
            elif board[max_row-row+direction[0]][col+direction[1]] != '':
                if not report_actions_left : stdio.writeln(ERRORS["field_not_free(r,c)"](row+direction[0],col+direction[1])); exit()
                else: return False
                
            
            # Move piece
            elif report_actions_left:
                return True
            else:
                board[max_row-row+direction[0]][col+direction[1]] = piece_type_raw
                board[max_row-row][col] = ""
            
            # If function in validation mode, return False since no moves were made
            if report_actions_left:
                return False
            else:
                return [None]

        # Moving B or C blocks
        elif piece_type in list("bc"):

            sizes = {'b' : 2, "c" : 3}
            directions = {'u' : (-1,0), 'd' : (1,0), 'l' : (0,-1), 'r' : (0,1)}

            direction = directions[move_type]
            size = sizes[piece_type]
            
            if piece_upright:

                if not check_coordinates_range_inclusive(max_row-row+direction[0]*size, col+direction[1]*size,0, max_row, 0, max_col): 
                    if not report_actions_left : stdio.writeln(ERRORS["beyond_board"]); exit()
                    else: return False

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
                    if report_actions_left:
                        return True
                    else:
                        board[max_row-row][col] = ''
                        return ['s', 2 if piece_type == "b" else 3]

                obstructed.sort(key=lambda x : x[0]+x[1])
                if not valid: 
                    if not report_actions_left : stdio.writeln(ERRORS["field_not_free(r,c)"](obstructed[0][0],obstructed[0][1])); exit()
                    else: return False

                if report_actions_left:
                    return True
                else:
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
                        if not check_range_inclusive(col+direction[1],0,max_col):
                            if not report_actions_left : stdio.writeln(ERRORS["beyond_board"]); exit()
                            else: return False

                        # Check destination
                        for i in range(size):
                            slot = board[max_row-row-i][col+direction[1]]
                            if slot != '': valid = False; obstructions.append((row+i,col+direction[1]))
                            elif slot != 's': all_sinks = False
                        
                        # Sink a piece
                        if all_sinks:
                            
                            # Clear piece and connected codes
                            for i in range(size):

                                if report_actions_left:
                                    return True
                                else:
                                    board[max_row-row-i][col] = ""
                                    return ['s', 2 if piece_type == "b" else 3]
                        
                        # Move piece
                        elif valid:
                            
                            if report_actions_left:
                                return True
                            else:

                                # Copy origin
                                board[max_row-row][col+direction[1]] = board[max_row-row][col]
                                board[max_row-row][col] = ''
                                new_code = f"{row*(max_col+1) + col + direction[1]}"

                                # clear piece and connected codes and copy to new destinations
                                for i in range(1,size):
                                    board[max_row-row-i][col] = ""
                                    board[max_row-row-i][col+direction[1]] = new_code
                        
                        # Obstructions
                        elif not report_actions_left :
                            stdio.writeln(ERRORS["field_not_free(r,c)"](obstructions[0][0],obstructions[0][1]))
                    

                    # Flip upright
                    else:

                        destination = None
                        destination_coordinates = (0,0)

                        if move_type == 'u':
                            if not check_range_inclusive(max_row-row-size,0,max_row): 
                                if not report_actions_left : stdio.writeln(ERRORS["beyond_board"]); exit()
                                else: return False

                            destination = board[max_row-row-size][col]
                            destination_coordinates = (max_row-row-size,col)

                        else:
                            if not check_range_inclusive(max_row-row+1,0,max_row): 
                                if not report_actions_left : stdio.writeln(ERRORS["beyond_board"]); exit()
                                else: return False

                            destination = board[max_row-row+1][col]
                            destination_coordinates = (max_row-row+1,col)

                        if destination == "s":

                            if report_actions_left:
                                return True
                            else:
                                for i in range(size):
                                    board[max_row-row-i][col] = ''
                                return ['s', 2 if piece_type == "b" else 3]

                        elif destination == '':

                            if report_actions_left:
                                return True
                            
                            else:
                                for i in range(size):
                                    board[max_row-row-i][col] = ''
                                board[destination_coordinates[0]][destination_coordinates[1]] = piece_type_raw
                            
                        else:
                            if not report_actions_left : stdio.writeln(ERRORS["field_not_free(r,c)"](max_row-destination_coordinates[0],destination_coordinates[1]))
                
                # Horizontally aligned
                else:

                    # Roll over
                    if move_type in list("ud"):

                        valid = True
                        all_sinks = True
                        obstructions = []

                        # Check if move on board
                        if not check_range_inclusive(max_row-row+direction[0], 0, max_row): 
                            if not report_actions_left : stdio.writeln(ERRORS["beyond_board"]); exit()
                            else: return False

                        # Validate destination
                        for i in range(size):
                            slot = board[max_row-row+direction[0]][col+i]
                            if slot != '': valid = False; obstructions.append((row-direction[0],col+i))
                            elif slot != 's': all_sinks = False

                        # Sink piece
                        if all_sinks:
                            if report_actions_left:
                                return True
                            else:
                                for i in range(size):
                                    board[max_row-row][col+i] = ""
                                return ['s', 2 if piece_type == "b" else 3]

                        # Move piece                        
                        elif valid:
                            
                            if report_actions_left:
                                return True
                            else:
                                # move origin and define new piece code
                                new_code = f"{(row-direction[0])*(max_col+1) + col}"
                                board[max_row-row+direction[0]][col] = board[max_row-row][col]
                                board[max_row-row][col] = ''

                                for i in range(1,size):
                                    board[max_row-row][col+i] = ''
                                    board[max_row-row+direction[0]][col+i] = new_code

                    
                    # Flip upright
                    else:

                        destination = None
                        destination_coordinates = (0,0)

                        if move_type == 'r':
                            if not check_range_inclusive(col+size,0,max_col): 
                                if not report_actions_left : stdio.writeln(ERRORS["beyond_board"]); exit()
                                else: return False

                            destination = board[max_row-row][col+size]
                            destination_coordinates = (max_row-row,col+size)

                        else:
                            if not check_range_inclusive(col-1,0,max_col): 
                                if not report_actions_left : stdio.writeln(ERRORS["beyond_board"]); exit()
                                else: return False

                            destination = board[max_row-row][col-1]
                            destination_coordinates = (max_row-row,col-1)

                        if destination == "s":
                            if report_actions_left:
                                return True
                            else:
                                for i in range(size):
                                    board[max_row-row][col+i] = ''
                                return ['s', 2 if piece_type == "b" else 3]

                        elif destination == '':

                            if report_actions_left:
                                return True
                            else:
                                for i in range(size):
                                    board[max_row-row][col+i] = ''
                                board[destination_coordinates[0]][destination_coordinates[1]] = piece_type_raw
                            
                        else:
                            if not report_actions_left : stdio.writeln(ERRORS["field_not_free(r,c)"](max_row-destination_coordinates[0],destination_coordinates[1]))

            if report_actions_left:
                return False
            else:
                return [None]

                    
        # Moving a 2x2x2 block
        elif piece_type == "d":

            # Validate second turn d piece moving
            if turn_number == 2: 
                if not report_actions_left : stdio.writeln(ERRORS["d_second_turn"]); exit()
                else: return False
            
            directions = {"u" : (-1,0), "d" : (1,0), "l" : (0,-1), "r" : (0,1)}
            direction = directions[move_type]

            # Check board
            valid = True
            for r in range(2):
                for c in range(2):
                    if not check_coordinates_range_inclusive(max_row-row-r+direction[0]*2,col+c+direction[1]*2,0,max_row,0,max_col): valid = False

            if not valid : 
                if not report_actions_left : stdio.writeln(ERRORS["beyond_board"]); exit()
                else: return False

            # Validate destination
            valid = True
            all_sinks = True

            destination_origin = (max_row-row+direction[0]*2, col+direction[1]*2)
            for r in range(2):
                for c in range(2):
                    if board[destination_origin[0]-r][destination_origin[1]+c] != '': valid = False
                    elif board[destination_origin[0]-r][destination_origin[1]+c] != 's': all_sinks = False

            # Sink piece
            if all_sinks:
                if report_actions_left:
                    return True
                else:
                    for r in range(2):
                        for c in range(2):
                            board[max_row-row-r][col+c] == ''
                    return ['s',4]
            
            # Move piece
            elif valid:

                if report_actions_left:
                    return True
                
                else:
                    # write new code
                    new_code = f"{(row-direction[0]*2)*(max_col+1) + (col+direction[1]*2)}"
                    for r in range(2):
                        for c in range(2):
                            board[destination_origin[0]-r][destination_origin[1]+c] = new_code
                    
                    # set new origin
                    board[destination_origin[0]][destination_origin[1]] = piece_type_raw

                    # clear old position
                    for r in range(2):
                        for c in range(2):
                            board[max_row-row-r][col+c] = ""

            if report_actions_left:
                return False
            else:
                return ["d"]

        elif piece_type == "s":
            
            # Find origin of sink
            if col > 0:
                if board[max_row-row][col-1] == "s": col -= 1
            if row > 0:
                if board[max_row-row+1][col] == "s": row -= 1

                    
            # get and validate remaining sink moves
            team = 'l' if lights_turn else 'd'
            moves_left = sink_moves[team]
            if moves_left == 0: 
                if not report_actions_left : stdio.writeln(ERRORS["no_sink_moves"]); exit()
                else: return False

            else: sink_moves[team] -= 1

            # Get sink size
            size = get_sink_size(board,row,col)

            # Check destination
            directions = {"u" : (-1,0), "d" : (1,0), "l" : (0,-1), "r" : (0,1)}
            direction = directions[move_type]
            destination = (max_row-row+direction[0]*size, col+direction[1]*size)

            # Check destination coordinates and occupancy
            valid = True
            free = True
            obstructions = []

            for r in range(size):
                for c in range(size):
                    if not check_coordinates_range_inclusive(destination[0]-r,destination[1]+c, 0, max_row, 0, max_col): valid = False
                    elif board[destination[0]-r][destination[1]+c] != '': free = False; obstructions.append((max_row-destination[0]+r, destination[1]+c))

            # Report errors and exit
            if not valid : 
                if not report_actions_left : stdio.writeln(ERRORS["beyond_board"]); exit()
                else: return False

            if not free : 
                if not report_actions_left : stdio.writeln(ERRORS["field_not_free(r,c)"](obstructions[0][0], obstructions[0][1])); exit()
                else: return False
            
            # Check if future position will have adjacent sinks
            future_board = copy_board(board)
            for r in range(size):
                for c in range(size):
                    future_board[max_row-row-r][col+c] = ''
            if not check_no_sink_adjacency(future_board, destination[0], destination[1], size): 
                if not report_actions_left : stdio.writeln(ERRORS["sink_adjacency"]); exit()
                else: return False

            # Clear current location
            if report_actions_left:
                return True
            else:
                for r in range(size):
                    for c in range(size):
                        board[max_row-row-r][col+c] = ''
                        board[destination[0]-r][destination[1]+c] = 's'

                return [None]


        # Valid piece not at coords given
        else:
            if not report_actions_left : stdio.writeln(ERRORS["no_piece(r,c)"](row,col)); exit()
            else: return False
        
    elif move_type == "f":

        # Check if any freezes remaining
        if freezes["l" if lights_turn else "d"] <= 0: 
            if not report_actions_left : stdio.writeln(ERRORS['no_freezes']); exit()
            else: return False

        # Check if piece at coordinates
        piece_type = board[max_row-row][col]

        # sink?
        if piece_type == "s": 
            if not report_actions_left : stdio.writeln(ERRORS["piece_not_owned"]); exit()
            else: return False
        
        # get team
        light_team = (lambda x: True if x.lower() == x else False)(board[max_row-row][col]) # True if piece owned by light team
        
        if light_team != lights_turn:
            freezes["l" if lights_turn else "d"] -= 1
            return ( "f", row, col, light_team )

        else:
            if not report_actions_left : stdio.writeln(ERRORS["piece_not_owned"]); exit()
            else: return False



    else:
        if not report_actions_left : stdio.writeln(ERRORS["invalid_direction(d)"](move_type)); exit()
        else: return False


# Game loop
def main( args ):


    # Board setup
    board = [["" for x in range(args["board_width"])] for y in range(args["board_height"])]
    read_stdin_setup_to_board(board)

    # Gameloop var
    lights_turn = True
    turn_counter = 0
    pre_turn_board = copy_board(board)
    sink_moves = {"l" : 2, "d" : 2}
    freezes = {"l" : 2, "d" : 2}
    player_scores = {"l" : 0, "d" : 0}

    frozen_pieces = [] # FMT (row, col, owned_by_light?, counter)

    while True:
        
        # Keep track of turns
        turn_counter += 1
        if turn_counter == 3: turn_counter = 1; lights_turn = not lights_turn
        
        # Track board before first turn
        if turn_counter == 1:
            pre_turn_board = copy_board(board)


        check_for_moves(board, lights_turn, turn_counter, sink_moves, frozen_pieces, freezes)
        # Read command or end partial game

        if stdio.hasNextLine():
            command = stdio.readLine()
        else:
            quit()

        # Check for possible moves
        result = move_pieces(board, command, lights_turn, turn_counter, sink_moves, frozen_pieces, freezes)


        # Ensure 2x2x2 movement ends turn
        if result[0] == "d":
            turn_counter = 2

        # Add frozen piece to tracker
        elif result[0] == "f":
            frozen_pieces.append((result[1], result[2], result[3], 2))
            turn_counter -= 1
            continue

        elif result[0] == "s":
            player_scores["l" if lights_turn else "d"] += result[1]

        # Check board after second turn
        elif turn_counter == 2:
            if board == pre_turn_board:
                stdio.writeln(ERRORS["repeated_position"])
                exit()
        
        print_board(board)
        check_win_conditions(board, player_scores, freezes, sink_moves)
        
        

        # Update frozen pieces
        for frozen_piece in frozen_pieces:
            if frozen_piece[2] == lights_turn:
                frozen_piece[3] -= 1

            if frozen_piece[3] == 0:
                frozen_pieces.remove(frozen_piece)
        
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
