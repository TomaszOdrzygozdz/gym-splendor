"""This file stores graphics configuration."""

from envs.mechanics.enums import GemColor, Row

# Dictionary linking gem colors to color of rendering.
color_dict_tkiter = {GemColor.RED: 'red', GemColor.GREEN: 'green', GemColor.BLUE: 'blue', GemColor.WHITE: 'white',
                     GemColor.BLACK: 'black',
                     GemColor.GOLD: 'gold'}

# Card visual settings:
CARD_BACKGROUND = 'white'
CARD_VICTORY_POINTS_FONT_COLOR = 'darkblue'
CARD_VICTORY_POINTS_FONT = 'Times 18 italic bold'
CARD_PRICE_FONT_COLOR = 'black'
CARD_PRICE_FONT = 'Times 14 italic bold'

#Noble visual settings:
NOBLE_BACKGROUND= 'white'
NOBLE_PRICE_FONT_COLOR = 'white'
NOBLE_PRICE_FONT = 'Arial 12 bold'

#Board visual settings:
BOARD_TITLE = 'Board'
BOARD_NAME_FONT = 'Arial 14 bold italic'
BOARD_NAME_FONT_COLOR = 'darkgreen'

#Placing cards on board setting:
POSITION_Y_DICT = {Row.CHEAP: 0, Row.MEDIUM: 1, Row.EXPENSIVE: 2} #This dictionary determines how cards would by placed
# in the board (values may be 0, 1, 2. Higher values means that rowe
#is closer to the bottom of the windom. Default is: {Row.CHEAP: 0, Row.MEDIUM: 1, Row.EXPENSIVE: 2}
HORIZONTAL_CARD_DISTANCE = 105
VERTICAL_CARD_DISTANCE =  145
HORIZONTAL_NOBLE_DISTANCE = 80
NOBLE_HORIZONTAL_SHIFT = 20 #determines where the set of nobles lies on the board
NOBLE_VERTICAL_SHIFT = 550 #determines where the set of nobles lies on the board

#time settings
#WINDOW_REFRESH_INTERVAL = 0.01

