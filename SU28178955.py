
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


VALIDATORS = {
    "range" : check_range_inclusive,
    "coordinates_range" : check_coordinates_range_inclusive,
}


def read_stdin_to_board( board ):

    while True:

        line = stdio.readLine()
        line_arguments = line.split(" ")


        if line_arguments[0] == "#":
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
            if not VALIDATORS["coordinates_range"](row, col, 0, args["board_height"]-1, 0, args["board_width"]-1):
                stdio.writeln(ERRORS['not_on_board(r,c)'](row,col))
                continue

            # Size check
            if size == 2:
                if not VALIDATORS["coordinates_range"](row, col, 0, args["board_height"]-2, 0, args["board_width"]-2):
                    stdio.writeln(ERRORS["sink_wrong_pos"])
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

            if not VALIDATORS["coordinates_range"](row, col, 0, args["board_height"]-1, 0, args["board_width"]-1):
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
            if not VALIDATORS["coordinates_range"](row, col, 0, args["board_height"]-1, 0, args["board_width"]-1):
                stdio.writeln(ERRORS['not_on_board(r,c)'](row,col))
                continue

            # Place pieces    
            if piece_type in ['a','b','c']:
                board[args["board_height"]-row-1][col] = id

            elif piece_type == 'd':

                # Size check
                if not VALIDATORS["coordinates_range"](row, col, 0, args["board_height"]-2, 0, args["board_width"]-2):
                    stdio.writeln(ERRORS["piece_wrong_pos"])
                    continue

                root_id = int(row)*args["board_width"]+int(col)
                board[args["board_height"]-row-1][col] = id
                board[args["board_height"]-row-2][col] = root_id
                board[args["board_height"]-row-2][col+1] = root_id
                board[args["board_height"]-row-1][col+1] = root_id
                
            else:
                stdio.writeln(ERRORS["invalid_piece(p)"](piece_type))

        else:
            stdio.writeln(ERRORS["invalid_object(o)"](line_arguments[0]))


def print_board( board ):

    height, width = (len(board), len(board[0]))

    # Top coordinates
    stdio.writeln(f"   {'  '.join([str(x) for x in range(width)])}")

    for rdx in range(height*2+1):

        line = ''

        if rdx % 2 == 0:
            line += f'  {"".join(["+--" for x in range(width)])}+'
        else:
            pass

        stdio.writeln(line)



def main( args ):

    board = [[" " for x in range(args["board_width"])] for y in range(args["board_height"])]
    read_stdin_to_board(board)
    print_board(board)




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
            if not ( VALIDATORS["range"](args["board_height"], 8, 10) and VALIDATORS["range"](args["board_height"], 8, 10) and args["gui"] in [0,1]):
                stdio.writeln(ERRORS['illegal'])
                valid_args = False

        except:
            stdio.writeln(ERRORS['illegal'])
            valid_args = False


        # Check validation and start
        if valid_args:
            main(args)

