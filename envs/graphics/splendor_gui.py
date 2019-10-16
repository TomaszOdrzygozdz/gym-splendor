from tkinter import *
from time import sleep

from pyglet.media import Player

from envs.mechanics.enums import Row
from envs.graphics.graphics_settings import *
from envs.mechanics.board import Board
from envs.mechanics.card import Card
from envs.mechanics.noble import Noble
from envs.mechanics.players_hand import PlayersHand


class SplendorGUI():
    """Class that is used to render the game."""

    def __init__(self):
        self.close_window = False
        self.main_window = Tk()
        self.main_window.title("Splendor kaioshin bakemono")
        self.main_window.geometry('1550x780')
        self.main_canvas = Canvas(self.main_window, width=1550, height=780)
        self.main_canvas.place(x=0, y=0)
        self.drawn_objects = set()

    def keep_window_open(self):
        mainloop()

    def draw_card(self,
                  card: Card,
                  x_coord: int,
                  y_coord: int,
                  draw_buy_button: bool = False,
                  draw_reserve_button: bool = False) -> None:

        """Draws a card in the main window.

        Parameters:
        _ _ _ _ _ _
        card: Card to draw.
        x_coord: Horizontal coordinate (from top left corner)
        y_coord: Vertical coordinate (from top left corner)
        draw_buy_button: Determines if create a buy action button associated with this card.
        draw_reserve_button: Determines if create a reserve action button associated with this card"""

        self.drawn_objects.add(
            self.main_canvas.create_rectangle(x_coord, y_coord, x_coord + CARD_WIDTH, y_coord + CARD_HEIGHT, fill=CARD_BACKGROUND))
        self.drawn_objects.add(
            self.main_canvas.create_text(x_coord + VICTORY_POINT_POSITION_X, y_coord + VICTORY_POINT_POSITION_Y,
                                         fill=CARD_VICTORY_POINTS_FONT_COLOR,
                                         font=CARD_VICTORY_POINTS_FONT, text=str(card.victory_points)))
        self.drawn_objects.add(
            self.main_canvas.create_rectangle(x_coord + PROFIT_BOX_POSITION_X, y_coord + PROFIT_BOX_POSITION_Y,
                                              x_coord + PROFIT_BOX_POSITION_X + PROFIT_BOX_SIZE,
                                              y_coord + PROFIT_BOX_POSITION_Y + PROFIT_BOX_SIZE,
                                              fill=color_dict_tkiter[card.discount_profit]))
        self.drawn_objects.add(self.main_canvas.create_line(x_coord + LINE_X, y_coord + LINE_Y,
                                                            x_coord + LINE_X + LINE_LENGTH, y_coord + LINE_Y))
        for color_index, color in enumerate(card.price.non_empty_stacks()):
            position_index = 4 - color_index
            self.drawn_objects.add(self.drawn_objects.add(self.main_canvas.create_oval(x_coord + PRICE_COIN_START_X,
                                                                                       y_coord + PRICE_COIN_START_Y +
                                                                                       PRICE_COIN_SHIFT * position_index,
                                                                                       x_coord + PRICE_COIN_START_X +
                                                                                       PRICE_COIN_SIZE,
                                                                                       y_coord + PRICE_COIN_START_Y +
                                                                                       PRICE_COIN_SIZE +
                                                                                       PRICE_COIN_SHIFT * position_index,
                                                                                       fill=color_dict_tkiter[color])))
            self.drawn_objects.add(
                self.main_canvas.create_text(x_coord + PRICE_VALUE_X, y_coord + PRICE_VALUE_Y +
                                             PRICE_COIN_SHIFT * position_index, font=CARD_PRICE_FONT,
                                             fill=CARD_PRICE_FONT_COLOR, text=str(card.price.value(color))))

    def draw_noble(self,
                   noble: Noble,
                   x_coord: int,
                   y_coord: int) -> None:
        """Draws a noble in the main window.

                Parameters:
                _ _ _ _ _ _
                card: Card to draw.
                x_coord: Horizontal coordinate (from top left corner)
                y_coord: Vertical coordinate (from top left corner)
                draw_buy_button: Determines if create a buy action button associated with this card.
                draw_reserve_button: Determines if create a reserve action button associated with this card"""

        self.drawn_objects.add(self.main_canvas.create_rectangle(x_coord, y_coord, x_coord + NOBLE_WIDTH,
                                                                 y_coord + NOBLE_HEIGHT))
        for position_index, color in enumerate(noble.price.non_empty_stacks()):
            position_x = x_coord + NOBLE_PRICE_BOX_X
            position_y = y_coord + NOBLE_PRICE_BOX_SHIFT * position_index + NOBLE_PRICE_BOX_Y
            self.drawn_objects.add(self.main_canvas.create_rectangle(position_x, position_y,
                                                                     position_x + NOBLE_PRICE_BOX_SIZE,
                                                                     position_y + NOBLE_PRICE_BOX_SIZE,
                                                                     fill=color_dict_tkiter[color]))
            self.drawn_objects.add(self.main_canvas.create_text(position_x + NOBLE_PRICE_VALUE_X,
                                                                position_y + NOBLE_PRICE_VALUE_Y, font=NOBLE_PRICE_FONT,
                                         fill=NOBLE_PRICE_FONT_COLOR, text=str(noble.price.value(color))))

    def draw_board(self,
                   board: Board,
                   x_coord: int,
                   y_coord: int,
                   active_players_hand: PlayersHand = None) -> None:

        """Draws the board, that is: cards that lie on the table, nobles that lie on the table and coins.
        Parameters:
        _ _ _ _ _ _
        board: Board to draw.
        x_coord: Horizontal ccordinate (from left top corner).
        y_coord: Vertical coordinate (from left top corner).
        active_players_hand: The hand of the player that is currently active. This argument is optional and is used to
        determine which cards should be given buy or reserve buttons. If the value is None no buttons are drawn."""

        self.drawn_objects.add(
            self.main_canvas.create_text(x_coord + 210, y_coord - 20, fill=BOARD_NAME_FONT_COLOR, text=BOARD_TITLE,
                                         font=BOARD_NAME_FONT))

        #dictionary used to keep track of drawn cards
        cards_already_drawn = {row : set() for row in Row}
        for card in board.cards_on_board:
            position_x = HORIZONTAL_CARD_DISTANCE * len(cards_already_drawn[card.row])
            cards_already_drawn[card.row].add(card)
            self.drawn_objects.add(self.draw_card(card, x_coord + position_x, y_coord +
                                                  VERTICAL_CARD_DISTANCE * POSITION_Y_DICT[card.row]))

        for position_index, noble_card in enumerate(board.nobles_on_board):
            position_x = NOBLES_START_X + x_coord + HORIZONTAL_NOBLE_DISTANCE * position_index
            position_y = NOBLES_START_Y + y_coord
            self.drawn_objects.add(self.draw_noble(noble_card, position_x, position_y))


    def draw_players_hand(self,
                          players_hand: PlayersHand,
                          x_ccord: int,
                          y_coord: int,
                          draw_buy_reserved_buttons: bool = False) -> None:
        """Draws a players hands in a given position.
        Parameters:
        _ _ _ _ _ _
        players_hand: A players hand to draw.
        x_coord: Horizontal coordinate (from left top corner).
        y_coord: Vertical coordinate (from left top corner).
        draw_reserved_buttons: Determines if draw action buy reserved button on reserved cards."""



