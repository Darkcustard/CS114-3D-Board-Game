
import stdio
import sys

ERRORS = {
    "illegal" : "ERROR: Illegal argument",
    "few_args" : "ERROR: Too few arguments",
    "many_args" : "ERROR: Too many arguments",
}



def read_stdin_to_board( board ):

    while True:

        line = stdio.readLine()
        line_arguments = line.split(" ")

        match line_arguments[0]:

            case "#":
                break
            
            # Place sinks
            case "s":
                _, size, row, col = line_arguments
                for r in range(int(size)):
                    for c in range(int(size)):
                        board[args["board_height"]-int(row)-r-1][int(col)+c] = "s"

            # Place blocked tiles
            case "x":
                _, row, col = line_arguments
                board[args["board_height"]-int(row)-1][int(col)] = "x"

            # Place Pieces
            case "d" | "l":

                team, piece_type, row, col = line_arguments

                # Choose board id
                if team == "d": id = piece_type.upper()
                else: id = piece_type
                

                # Place pieces    
                if piece_type in ['a','b','c']:
                    board[args["board_height"]-int(row)-1][int(col)] = id

                elif piece_type == 'd':
                    root_id = int(row)*args["board_width"]+int(col)
                    board[args["board_height"]-int(row)-1][int(col)] = id
                    board[args["board_height"]-int(row)-2][int(col)] = root_id
                    board[args["board_height"]-int(row)-2][int(col)+1] = root_id
                    board[args["board_height"]-int(row)-1][int(col)+1] = root_id
                    
                else:
                    stdio.writeln(ERRORS["illegal"])

            case _:
                stdio.writeln(ERRORS['illegal'])


def main( args ):

    board = [[" " for x in range(args["board_width"])] for y in range(args["board_height"])]
    read_stdin_to_board(board)
   

    for row in board:
        stdio.writeln(row)




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
        except:
            stdio.writeln(ERRORS['illegal'])
            valid_args = False

        # Range Check
        if not (args["board_height"] >= 8 and args["board_height"] <= 10 and args["board_width"] >= 8 and args["board_width"] <= 10 and args["gui"] in [0,1]):
            stdio.writeln(ERRORS['illegal'])
            valid_args = False
       
        # Check validation and start
        if valid_args:
            main(args)

