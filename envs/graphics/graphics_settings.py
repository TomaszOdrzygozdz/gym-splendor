"""This file stores graphics configuration."""

from envs.mechanics.enums import GemColor

# Dictionary linking gem colors to color of rendering.
color_dict_tkiter = {GemColor.RED: 'red', GemColor.GREEN: 'green', GemColor.BLUE: 'blue', GemColor.WHITE: 'white',
                     GemColor.BLACK: 'black',
                     GemColor.GOLD: 'gold'}

# Card visual settings
CARD_BACKGROUND = 'white'
CARD_FONT_VICTORY_POINTS_COLOR = 'darkblue'
CARD_FONT_VICTORY_POINTS = 'Times 18 italic bold'
CARD_FONT_PRICE_COLOR = 'black'
CARD_FONT_PRICE = 'Times 14 italic bold'
