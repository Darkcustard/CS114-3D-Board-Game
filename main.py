
import stdio
import stddraw
import sys
import math



def main( GUI ):
    
    if GUI:

        # Window init
        WINDOW = stddraw.pygame.display.set_mode((500,500))
        stddraw.pygame.display.set_caption("3D Boardgame")

        # Var
        RUNNING = True

        while RUNNING:

            # Check for quit conditions
            for event in stddraw.pygame.event.get():
                if event.type == stddraw.pygame.QUIT:
                    RUNNING = False



            WINDOW.fill((255,255,255))
            stddraw.pygame.display.update()


    else:

        stdio.writeln("NO GUI MODE SELECTED.")



# Program Entry point
if __name__ == "__main__":
    
    # CL Input and validation
    arguments = sys.argv

    if len(arguments) == 2:

        _, GUI = arguments
        main(GUI)

    else:
        stdio.writeln(f"ERROR: expected exactly 1 argument, found {len(arguments)-1} arguments.")