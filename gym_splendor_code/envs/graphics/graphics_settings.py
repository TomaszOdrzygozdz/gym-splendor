"""This file stores graphics configuration."""

from gym_splendor_code.envs.mechanics.enums import GemColor, Row

# Dictionary linking gem colors to color of rendering.
color_dict_tkiter = {GemColor.RED: 'red', GemColor.GREEN: 'green', GemColor.BLUE: 'blue', GemColor.WHITE: 'white',
                     GemColor.BLACK: 'black',
                     GemColor.GOLD: 'gold'}

#scaling_factor:
SCALING_FACTOR = 1
WINDOW_TITLE = 'Splendor kayoshin bakemono'
WINDOW_GEOMETRY = '1550x780'
WINDOW_REFRESH_TIME = 0.01 #time in seconds
GAME_SPEED = 5.5 #time between consecutive frames of game
GAME_INITIAL_DELAY = 1 #time before teh game starts (needed for example to start recording)

# Card visual settings:
#____________________________________________________________
CARD_BACKGROUND = 'white'
CARD_VICTORY_POINTS_FONT_COLOR = 'darkblue'
CARD_VICTORY_POINTS_FONT = 'Times {} italic bold'.format(int(18*SCALING_FACTOR))
CARD_PRICE_FONT_COLOR = 'black'
CARD_PRICE_FONT = 'Times {} italic bold'.format(int(14*SCALING_FACTOR))
CARD_NAME_FONT = 'Times {} italic'.format(int(12*SCALING_FACTOR))
CARD_NAME_COLOR = 'black'

CARD_WIDTH = 100*SCALING_FACTOR
CARD_HEIGHT = 140*SCALING_FACTOR

CARD_NAME_POSITION_X = 75*SCALING_FACTOR
CARD_NAME_POSITION_Y = 130*SCALING_FACTOR

VICTORY_POINT_POSITION_X = 15*SCALING_FACTOR
VICTORY_POINT_POSITION_Y = 15*SCALING_FACTOR
PROFIT_BOX_POSITION_X = 70*SCALING_FACTOR
PROFIT_BOX_POSITION_Y = 5*SCALING_FACTOR
PROFIT_BOX_SIZE = 20*SCALING_FACTOR
LINE_X = 5*SCALING_FACTOR
LINE_Y = 30*SCALING_FACTOR
LINE_LENGTH = 85*SCALING_FACTOR

PRICE_COIN_START_X = 5*SCALING_FACTOR
PRICE_COIN_START_Y = 40*SCALING_FACTOR
PRICE_COIN_SHIFT = 20*SCALING_FACTOR
PRICE_COIN_SIZE = 15*SCALING_FACTOR
PRICE_VALUE_X = 28*SCALING_FACTOR
PRICE_VALUE_Y = 48*SCALING_FACTOR

BUY_BUTTON_TITLE = '  Buy   '
BUY_BUTTON_FONT = 'Arial {} bold'.format(int(7*SCALING_FACTOR))
BUY_BUTTON_X = 43*SCALING_FACTOR
BUY_BUTTON_Y = 100*SCALING_FACTOR

RESERVE_BUTTON_TITLE = ' Reserve '
RESERVE_BUTTON_FONT = 'Arial {} bold'.format(int(7*SCALING_FACTOR))
RESERVE_BUTTON_X = 43*SCALING_FACTOR
RESERVE_BUTTON_Y = 70*SCALING_FACTOR


#Noble visual settings:
#____________________________________________________________
NOBLE_BACKGROUND= 'white'
NOBLE_PRICE_FONT_COLOR = 'black'
NOBLE_PRICE_FONT = 'Arial {} bold'.format(int(12*SCALING_FACTOR))

NOBLE_WIDTH = 70*SCALING_FACTOR
NOBLE_HEIGHT = 75*SCALING_FACTOR

NOBLE_PRICE_BOX_X = 10*SCALING_FACTOR
NOBLE_PRICE_BOX_Y = 5*SCALING_FACTOR
NOBLE_PRICE_BOX_SHIFT = 25*SCALING_FACTOR
NOBLE_PRICE_BOX_SIZE = 15*SCALING_FACTOR
NOBLE_PRICE_VALUE_X = 25*SCALING_FACTOR
NOBLE_PRICE_VALUE_Y = 5*SCALING_FACTOR


#Board visual settings:
#____________________________________________________________
BOARD_TITLE = 'Board'
BOARD_TITLE_POSITION_X = 210*SCALING_FACTOR
BOARD_TITLE_POSITION_Y = -20*SCALING_FACTOR
BOARD_NAME_FONT = 'Arial {} bold italic'.format(int(14*SCALING_FACTOR))
BOARD_NAME_FONT_COLOR = 'darkgreen'

#Placing cards on board settings:
POSITION_Y_DICT = {Row.CHEAP: 0, Row.MEDIUM: 1, Row.EXPENSIVE: 2} #This dictionary determines how cards would by placed
# in the board (values may be 0, 1, 2. Higher values means that rowe
#is closer to the bottom of the windom. Default is: {Row.CHEAP: 0, Row.MEDIUM: 1, Row.EXPENSIVE: 2}
HORIZONTAL_CARD_DISTANCE = 105*SCALING_FACTOR
VERTICAL_CARD_DISTANCE =  145*SCALING_FACTOR
HORIZONTAL_NOBLE_DISTANCE = 80*SCALING_FACTOR
NOBLE_HORIZONTAL_SHIFT = 20*SCALING_FACTOR #determines where the set of nobles lies on the board
NOBLE_VERTICAL_SHIFT = 550*SCALING_FACTOR #determines where the set of nobles lies on the board

NOBLES_START_X = 60*SCALING_FACTOR
NOBLES_START_Y = 550*SCALING_FACTOR

GEMS_BOARD_SHIFT = 70*SCALING_FACTOR
GEM_BOARD_OVAL_SIZE = 20*SCALING_FACTOR
GEMS_BOARD_VALUE_SHIFT = 10*SCALING_FACTOR
GEMS_BOARD_VALUE_VERTICAL_SHIFT = 30*SCALING_FACTOR
GEMS_BOARD_FONT = "Arial {} bold".format(int(10*SCALING_FACTOR))
GEMS_BOARD_X = 15*SCALING_FACTOR
GEMS_BOARD_Y = 450*SCALING_FACTOR

GEMS_SUMMARY_X = -25*SCALING_FACTOR
GEMS_SUMMARY_Y = 5*SCALING_FACTOR
GEMS_SUMMARY_TITLE = 'Gems '
GEMS_SUMMARY_FONT = 'Arial {} bold'.format(int(11*SCALING_FACTOR))

GEMS_ENTRY_INITIAL_X = 40*SCALING_FACTOR
GEMS_ENTRY_INITIAL_Y = 450*SCALING_FACTOR
GEM_ENTRY_SHIFT = 70*SCALING_FACTOR
GEM_ENTRY_WIDTH = 20*SCALING_FACTOR

TRADE_BUTTON_TITLE = 'Trade gems'
TRADE_BUTTON_FONT = 'Arial {} bold'.format(int(14*SCALING_FACTOR))
TRADE_BUTTON_X = -5*SCALING_FACTOR
TRADE_BUTTON_Y = 500*SCALING_FACTOR

CONFIRM_BUY_TITLE = 'Confirm buy'
CONFIRM_BUY_FONT = 'Arial {} bold'.format(int(14*SCALING_FACTOR))
CONFIRM_BUY_X = 130*SCALING_FACTOR
CONFIRM_BUY_Y = 500*SCALING_FACTOR

CONFIRM_RESERVE_TITLE = 'Confirm reservation'
CONFIRM_RESERVE_FONT = 'Arial {} bold'.format(int(14*SCALING_FACTOR))
CONFIRM_RESERVE_X = 270*SCALING_FACTOR
CONFIRM_RESERVE_Y = 500*SCALING_FACTOR


#Players hand visual settings:
#____________________________________________________________
#Placing cards settings:

PLAYERS_HAND_INITIAL_X = -80*SCALING_FACTOR
PLAYERS_HAND_INITIAL_Y = 60*SCALING_FACTOR
PLAYERS_HAND_HORIZONTAL_SHIFT = 105*SCALING_FACTOR
PLAYERS_HAND_VERTICAL_SHIFT = 30*SCALING_FACTOR

PLAYERS_NAME_X = 0*SCALING_FACTOR
PLAYERS_NAME_Y = 0*SCALING_FACTOR
PLAYERS_NAME_FONT_ACTIVE = 'Arial {} bold italic'.format(int(14*SCALING_FACTOR))
PLAYERS_NAME_FONT = 'Arial {} italic'.format(int(13*SCALING_FACTOR))

PLAYERS_POINTS_X = 190
PLAYERS_POINTS_Y = 0
PLAYERS_POINTS_TITLE = 'Points: '
PLAYERS_POINTS_FONT = 'Arial {}.format(int(13*SCALING_FACTOR))'

RESERVED_CARDS_HORIZONTAL_SHIFT = 105*SCALING_FACTOR
RESERVED_CARDS_INITIAL_X = 15*SCALING_FACTOR
RESERVED_CARDS_INITIAL_Y = 510*SCALING_FACTOR
RESERVED_CARDS_FONT = "Arial {} bold italic".format(int(10*SCALING_FACTOR))
RESERVED_CARDS_TITLE_X = 55*SCALING_FACTOR
RESERVED_CARDS_TITLE_Y = 490*SCALING_FACTOR
RESERVED_CARDS_Y = 510*SCALING_FACTOR
RESERVED_RECTANGLE_LEFT_TOP_X = 5*SCALING_FACTOR
RESERVED_RECTANGLE_LEFT_TOP_Y = 500*SCALING_FACTOR
RESERVED_RECTANGLE_RIGHT_BOTTOM_X = 360*SCALING_FACTOR
RESERVED_RECTANGLE_RIGHT_BOTTOM_Y = 665*SCALING_FACTOR
RESERVED_RECTANGLE_OUTLINE = 'red'

PLAYERS_HAND_GEMS_X = 10*SCALING_FACTOR
PLAYERS_HAND_GEMS_Y = 15*SCALING_FACTOR

PLAYERS_NOBLES_X = 400
PLAYERS_NOBLES_Y = 500
PLAYERS_NOBLES_SHIFT = 80

#State visual settings:
#____________________________________________________________
STATE_PLAYER_HORIZONTAL_SHIFT = 985*SCALING_FACTOR
STATE_PLAYER_VERTICAL_SHIFT = 500*SCALING_FACTOR
STATE_PLAYERS_X = 90*SCALING_FACTOR
STATE_PLAYERS_Y = 20*SCALING_FACTOR

STATE_BOARD_X = 550*SCALING_FACTOR
STATE_BOARD_Y = 100*SCALING_FACTOR

WARNING_FONT = 'Arial {} bold'.format(int(10*SCALING_FACTOR))
WARNING_COLOR = 'red'
WARNING_X = 450*SCALING_FACTOR
WARNING_Y = 742*SCALING_FACTOR

ACTION_FONT = 'Arial {} bold'.format(int(10*SCALING_FACTOR))
ACTION_COLOR = 'green'
ACTION_X = 750*SCALING_FACTOR
ACTION_Y = 765*SCALING_FACTOR


