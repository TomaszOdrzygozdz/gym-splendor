from tkinter import *

from gym_splendor_code.envs.mechanics.game_settings import *
from gym_splendor_code.envs.graphics.graphics_settings import *
from gym_splendor_code.envs.mechanics.action import ActionBuyCard, Action, ActionTradeGems, ActionReserveCard
from gym_splendor_code.envs.mechanics.board import Board
from gym_splendor_code.envs.mechanics.card import Card
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.noble import Noble
from gym_splendor_code.envs.mechanics.players_hand import PlayersHand
from gym_splendor_code.envs.mechanics.state import State

class SplendorGUI():
    """Class that is used to render the game."""

    def __init__(self, interactive=True):
        self.close_window = False
        self.main_window = Tk()
        self.main_window.title(WINDOW_TITLE)
        self.main_window.geometry(WINDOW_GEOMETRY)
        self.main_canvas = Canvas(self.main_window, width=1550, height=780)
        self.main_canvas.place(x=0, y=0)
        self.drawn_buttons = set()
        self.actual_action = None
        self.interactive = interactive
        self.board_x_ccord = None
        self.board_y_ccord = None
        self.entries = {}

    def set_action(self, action: Action) -> None:
        self.actual_action = action

    def reset_action(self) -> None:
        self.actual_action = None

    def read_action(self) -> Action:
        while self.actual_action is None:
            self.main_window.update_idletasks()
            self.main_window.update()
            self.main_window.after(int(WINDOW_REFRESH_TIME * 1000))
        action_to_return = self.actual_action
        self.reset_action()
        return action_to_return

    def keep_window_open(self):
        mainloop()

    def draw_card(self,
                  card: Card,
                  x_coord: int,
                  y_coord: int,
                  draw_buy_button: bool,
                  draw_reserve_button: bool,
                  state: State) -> None:

        """Draws a card in the main window.

        Parameters:
        _ _ _ _ _ _
        card: Card to draw.
        x_coord: Horizontal coordinate (from top left corner)
        y_coord: Vertical coordinate (from top left corner)
        draw_buy_button: Determines if create a buy action button associated with this card.
        draw_reserve_button: Determines if create a reserve action button associated with this card"""

        self.main_canvas.create_rectangle(x_coord, y_coord, x_coord + CARD_WIDTH, y_coord + CARD_HEIGHT,
                                          fill=CARD_BACKGROUND)
        self.main_canvas.create_text(x_coord + VICTORY_POINT_POSITION_X, y_coord + VICTORY_POINT_POSITION_Y,
                                     fill=CARD_VICTORY_POINTS_FONT_COLOR,
                                     font=CARD_VICTORY_POINTS_FONT, text=str(card.victory_points))

        self.main_canvas.create_rectangle(x_coord + PROFIT_BOX_POSITION_X, y_coord + PROFIT_BOX_POSITION_Y,
                                          x_coord + PROFIT_BOX_POSITION_X + PROFIT_BOX_SIZE,
                                          y_coord + PROFIT_BOX_POSITION_Y + PROFIT_BOX_SIZE,
                                          fill=color_dict_tkiter[card.discount_profit])
        self.main_canvas.create_line(x_coord + LINE_X, y_coord + LINE_Y,
                                     x_coord + LINE_X + LINE_LENGTH, y_coord + LINE_Y)
        for color_index, color in enumerate(card.price.non_empty_stacks()):
            position_index = 4 - color_index
            self.main_canvas.create_oval(x_coord + PRICE_COIN_START_X,
                                         y_coord + PRICE_COIN_START_Y +
                                         PRICE_COIN_SHIFT * position_index,
                                         x_coord + PRICE_COIN_START_X +
                                         PRICE_COIN_SIZE,
                                         y_coord + PRICE_COIN_START_Y +
                                         PRICE_COIN_SIZE +
                                         PRICE_COIN_SHIFT * position_index,
                                         fill=color_dict_tkiter[color])

            self.main_canvas.create_text(x_coord + PRICE_VALUE_X, y_coord + PRICE_VALUE_Y +
                                         PRICE_COIN_SHIFT * position_index, font=CARD_PRICE_FONT,
                                         fill=CARD_PRICE_FONT_COLOR, text=str(card.price.value(color)))

            self.main_canvas.create_text(x_coord + CARD_NAME_POSITION_X, y_coord + CARD_NAME_POSITION_Y,
                                         text=card.name, fill=CARD_NAME_COLOR, font=CARD_NAME_FONT)

        if draw_buy_button:
            buy_button = Button(self.main_window, font=BUY_BUTTON_FONT, text=BUY_BUTTON_TITLE,
                                command=lambda: self.prepare_to_buy(card, state))
            buy_button.place(x=x_coord + BUY_BUTTON_X, y=y_coord + BUY_BUTTON_Y)
            self.drawn_buttons.add(buy_button)

        if draw_reserve_button:
            reserve_button = Button(self.main_window, font=RESERVE_BUTTON_FONT, text=RESERVE_BUTTON_TITLE,
                                    command=lambda: self.prepare_to_reserve(card, state))
            reserve_button.place(x=x_coord + RESERVE_BUTTON_X, y=y_coord + RESERVE_BUTTON_Y)
            self.drawn_buttons.add(reserve_button)

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

        self.main_canvas.create_rectangle(x_coord, y_coord, x_coord + NOBLE_WIDTH,
                                          y_coord + NOBLE_HEIGHT)
        for position_index, color in enumerate(noble.price.non_empty_stacks()):
            position_x = x_coord + NOBLE_PRICE_BOX_X
            position_y = y_coord + NOBLE_PRICE_BOX_SHIFT * position_index + NOBLE_PRICE_BOX_Y
            self.main_canvas.create_rectangle(position_x, position_y,
                                              position_x + NOBLE_PRICE_BOX_SIZE,
                                              position_y + NOBLE_PRICE_BOX_SIZE,
                                              fill=color_dict_tkiter[color])
            self.main_canvas.create_text(position_x + NOBLE_PRICE_VALUE_X,
                                         position_y + NOBLE_PRICE_VALUE_Y, font=NOBLE_PRICE_FONT,
                                         fill=NOBLE_PRICE_FONT_COLOR,
                                         text=str(noble.price.value(color)))

    def draw_board(self,
                   board: Board,
                   x_coord: int,
                   y_coord: int,
                   state: State) -> None:

        """Draws the board, that is: cards that lie on the table, nobles that lie on the table and coins.
        Parameters:
        _ _ _ _ _ _
        board: Board to draw.
        x_coord: Horizontal coordinate (from left top corner).
        y_coord: Vertical coordinate (from left top corner).
        active_players_hand: The hand of the player that is currently active. This argument is optional and is used to
        determine which cards should be given buy or reserve buttons. If the value is None no buttons are drawn."""

        self.board_x_ccord = x_coord
        self.board_y_ccord = y_coord

        self.main_canvas.create_text(x_coord + BOARD_TITLE_POSITION_X, y_coord + BOARD_TITLE_POSITION_Y,
                                     fill=BOARD_NAME_FONT_COLOR, text=BOARD_TITLE, font=BOARD_NAME_FONT)

        # dictionary used to keep track of drawn cards
        cards_already_drawn = {row: set() for row in Row}
        for card in board.cards_on_board:
            position_x = HORIZONTAL_CARD_DISTANCE * len(cards_already_drawn[card.row])
            cards_already_drawn[card.row].add(card)
            self.draw_card(card, x_coord + position_x, y_coord + VERTICAL_CARD_DISTANCE * POSITION_Y_DICT[card.row],
                           state.active_players_hand().can_afford_card(card),
                           state.active_players_hand().can_reserve_card(), state)

        for position_index, noble_card in enumerate(board.nobles_on_board):
            position_x = NOBLES_START_X + x_coord + HORIZONTAL_NOBLE_DISTANCE * position_index
            position_y = NOBLES_START_Y + y_coord
            self.draw_noble(noble_card, position_x, position_y)

        self.draw_gems(board.gems_on_board, x_coord + GEMS_BOARD_X, y_coord + GEMS_BOARD_Y)

        if self.interactive:
            for gem_color in GemColor:
                gem_entry = Entry(self.main_window)
                gem_entry.place(x=x_coord + GEM_ENTRY_SHIFT * gem_color.value + GEMS_ENTRY_INITIAL_X,
                                y=y_coord + GEMS_ENTRY_INITIAL_Y, width=GEM_ENTRY_WIDTH)
                self.entries[gem_color] = gem_entry
                self.drawn_buttons.add(gem_entry)
            self.set_entries(GemsCollection())
            trade_button = Button(text=TRADE_BUTTON_TITLE, font=TRADE_BUTTON_FONT,
                                  command=lambda: self.set_action(ActionTradeGems(self.read_entries())))
            trade_button.place(x=x_coord + TRADE_BUTTON_X, y=y_coord + TRADE_BUTTON_Y)
            self.drawn_buttons.add(trade_button)

    def draw_gems(self,
                  gems_collection: GemsCollection,
                  x_coord: int,
                  y_coord: int) -> None:

        self.main_canvas.create_text(x_coord + GEMS_SUMMARY_X, y_coord + GEMS_SUMMARY_Y,
                                     text=GEMS_SUMMARY_TITLE + str(gems_collection.sum()), font=GEMS_SUMMARY_FONT)
        for gem_color in GemColor:
            self.main_canvas.create_oval(x_coord + GEMS_BOARD_SHIFT * gem_color.value,
                                         y_coord,
                                         x_coord + GEMS_BOARD_SHIFT * gem_color.value + GEM_BOARD_OVAL_SIZE,
                                         y_coord + GEM_BOARD_OVAL_SIZE,
                                         fill=color_dict_tkiter[gem_color])

            self.main_canvas.create_text(x_coord + GEMS_BOARD_SHIFT * gem_color.value +
                                         GEMS_BOARD_VALUE_SHIFT,
                                         y_coord + GEMS_BOARD_VALUE_VERTICAL_SHIFT,
                                         text=str(gems_collection.value(gem_color)),
                                         font=GEMS_BOARD_FONT)

    def draw_players_hand(self,
                          players_hand: PlayersHand,
                          x_coord: int,
                          y_coord: int,
                          active: bool,
                          state: State) -> None:
        """Draws a players hands in a given position.
        Parameters:
        _ _ _ _ _ _
        players_hand: A players hand to draw.
        x_coord: Horizontal coordinate (from left top corner).
        y_coord: Vertical coordinate (from left top corner).
        draw_reserved_buttons: Determines if draw action buy reserved button on reserved cards."""

        if active:
            players_name_font = PLAYERS_NAME_FONT_ACTIVE
        else:
            players_name_font = PLAYERS_NAME_FONT

        self.main_canvas.create_text(x_coord + PLAYERS_NAME_X, y_coord + PLAYERS_NAME_Y,
                                     text=players_hand.name, font=players_name_font)

        card_position_x_dict = {gem_color: gem_color.value * PLAYERS_HAND_HORIZONTAL_SHIFT for gem_color in GemColor if
                                gem_color != GemColor.GOLD}

        cards_presented = {gem_color: set() for gem_color in GemColor}
        # Draw all cards:
        for card in players_hand.cards_possessed:
            card_x_coord = PLAYERS_HAND_INITIAL_X + card_position_x_dict[card.discount_profit]
            card_y_coord = PLAYERS_HAND_INITIAL_Y + len(
                cards_presented[card.discount_profit]) * PLAYERS_HAND_VERTICAL_SHIFT
            self.draw_card(card, x_coord + card_x_coord, y_coord + card_y_coord, False, False, state)
            cards_presented[card.discount_profit].add(card)
        # Draw all reserved cards:
        self.main_canvas.create_text(x_coord + RESERVED_CARDS_TITLE_X,
                                     y_coord + RESERVED_CARDS_TITLE_Y, text="Reserved cards:",
                                     font=RESERVED_CARDS_FONT)
        self.main_canvas.create_rectangle(x_coord + RESERVED_RECTANGLE_LEFT_TOP_X,
                                          y_coord + RESERVED_RECTANGLE_LEFT_TOP_Y,
                                          x_coord + RESERVED_RECTANGLE_RIGHT_BOTTOM_X,
                                          y_coord + RESERVED_RECTANGLE_RIGHT_BOTTOM_Y,
                                          outline=RESERVED_RECTANGLE_OUTLINE)
        reserved_cards_presented = set()
        for card in players_hand.cards_reserved:
            card_x_coord = RESERVED_CARDS_INITIAL_X + len(reserved_cards_presented) * RESERVED_CARDS_HORIZONTAL_SHIFT
            card_y_coord = RESERVED_CARDS_INITIAL_Y
            self.draw_card(card, x_coord + card_x_coord, y_coord + card_y_coord, players_hand.can_afford_card(card) and
                           active, False, state)
            reserved_cards_presented.add(card)

        # Draw gems possessed by the player:
        self.draw_gems(players_hand.gems_possessed, x_coord + PLAYERS_HAND_GEMS_X, y_coord + PLAYERS_HAND_GEMS_Y)

        #Draw nobles possessed by the player

    def draw_state(self, state: State) -> None:
        """Draws the state. """
        for number, player in enumerate(state.list_of_players_hands):
            x_coord_player = STATE_PLAYERS_X + number % 2 * STATE_PLAYER_HORIZONTAL_SHIFT
            y_coord_player = STATE_PLAYERS_Y + (number - number % 2) / 2 * STATE_PLAYER_VERTICAL_SHIFT
            self.draw_players_hand(player, x_coord_player, y_coord_player, number == state.active_player_id, state)

        self.draw_board(state.board, STATE_BOARD_X, STATE_BOARD_Y, state)

    def prepare_to_buy(self,
                       card: Card,
                       state: State):

        price_after_discount = card.price % state.active_players_hand().discount()
        min_gold = (price_after_discount % state.active_players_hand().gems_possessed).sum()
        min_gold_price = GemsCollection({gem_color: min(price_after_discount.value(gem_color),
                                                        state.active_players_hand().gems_possessed.value(gem_color))
                                         for gem_color in GemColor})
        min_gold_price.gems_dict[GemColor.GOLD] = min_gold
        self.set_entries(min_gold_price)

        confirm_buy_button = Button(text=CONFIRM_BUY_TITLE, font=CONFIRM_BUY_FONT,
                                    command=lambda: self.do_buy(card, state))
        confirm_buy_button.place(x=self.board_x_ccord + CONFIRM_BUY_X, y=self.board_y_ccord + CONFIRM_BUY_Y)
        self.drawn_buttons.add(confirm_buy_button)

    def prepare_to_reserve(self,
                           card,
                           state: State):

        basic_gems_transfer = GemsCollection()
        if state.active_players_hand().gems_possessed.sum() < MAX_GEMS_ON_HAND and \
                state.board.gems_on_board.gems_dict[GemColor.GOLD] > 0:
            basic_gems_transfer.gems_dict[GemColor.GOLD] = 1
        self.set_entries(basic_gems_transfer)

        confirm_reserve_button = Button(text=CONFIRM_RESERVE_TITLE, font=CONFIRM_RESERVE_FONT,
                                        command=lambda: self.do_reserve(card, state))
        confirm_reserve_button.place(x=self.board_x_ccord + CONFIRM_RESERVE_X, y=self.board_y_ccord + CONFIRM_RESERVE_Y)
        self.drawn_buttons.add(confirm_reserve_button)

    def read_entries(self) -> GemsCollection:
        return GemsCollection({gem_color: int(self.entries[gem_color].get()) for gem_color in GemColor})

    def set_entries(self, gems_collection: GemsCollection) -> None:
        for gem_color in GemColor:
            self.entries[gem_color].delete(0, END)
            self.entries[gem_color].insert(0, gems_collection.value(gem_color))

    def clear_all(self):
        self.main_canvas.delete('all')
        for drawn_object in self.drawn_buttons:
            drawn_object.destroy()

    def show_warning(self, action):
        self.main_canvas.create_text(WARNING_X, WARNING_Y, text='{} is illegal.'.format(action),
                                     font=WARNING_FONT, fill=WARNING_COLOR)

    def show_last_action(self, action):
        self.main_canvas.create_text(ACTION_X, ACTION_Y, text='Last action: {}.'.format(action),
                                     font=ACTION_FONT, fill=ACTION_COLOR)

    def do_buy(self, card, state):
        price_after_discount = card.price % state.active_players_hand().discount()
        what_I_pay = self.read_entries()
        gold_to_use = what_I_pay.value(GemColor.GOLD)
        use_gold_as = price_after_discount - what_I_pay
        use_gold_as.gems_dict[GemColor.GOLD] = 0
        self.set_action(ActionBuyCard(card, gold_to_use, use_gold_as))

    def do_reserve(self, card, state):
        return_gem_color = None
        gems_transfer = self.read_entries()
        for gem_color in GemColor:
            if gems_transfer.gems_dict[gem_color] > 0 and gem_color != GemColor.GOLD:
                return_gem_color = gem_color
                break

        self.set_action(ActionReserveCard(card, gems_transfer.gems_dict[GemColor.GOLD] == 1, return_gem_color))
