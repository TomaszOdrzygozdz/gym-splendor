from tkinter import *
from time import sleep

from envs.graphics.graphics_settings import *
from envs.mechanics.card import Card

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

        """Draw a card in the main window.

        Parameters:
        _ _ _ _ _ _
        card: Card to draw.
        x_coord: Horizontal coordinate (from top left corner)
        y_coord: Vertical coordinate (from top left corner)
        draw_buy_button: Determines if create a buy action button associated with this card.
        draw_reserve_button: Determines if create a reserve action button associated with this card"""

        self.drawn_objects.add(
            self.main_canvas.create_rectangle(x_coord, y_coord, x_coord + 100, y_coord + 140, fill=CARD_BACKGROUND))
        self.drawn_objects.add(
            self.main_canvas.create_text(x_coord + 15, y_coord + 15, fill=CARD_FONT_VICTORY_POINTS_COLOR,
                                         font=CARD_FONT_VICTORY_POINTS, text=str(card.victory_points)))
        self.drawn_objects.add(
            self.main_canvas.create_rectangle(x_coord + 70, y_coord + 5, x_coord + 90, y_coord + 25,
                                              fill=
                                              color_dict_tkiter[card.discount_profit]))
        self.drawn_objects.add(self.main_canvas.create_line(x_coord + 5, y_coord + 30, x_coord + 95, y_coord + 30))
        for color_index, color in enumerate(card.price.non_empty_stacks()):
            position_index = 4 - color_index
            self.drawn_objects.add(self.drawn_objects.add(self.main_canvas.create_oval(x_coord + 5, y_coord + 40 +
                                                20 * position_index, x_coord + 20, y_coord + 55 + 20 * position_index,
                                                fill=color_dict_tkiter[color])))
            self.drawn_objects.add(
                self.main_canvas.create_text(x_coord + 28, y_coord + 48 + 20 * position_index, font=CARD_FONT_PRICE,
                                             fill=CARD_FONT_PRICE_COLOR, text=str(card.price.value(color))))

    