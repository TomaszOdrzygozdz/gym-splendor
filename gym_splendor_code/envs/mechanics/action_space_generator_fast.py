from typing import List
from itertools import combinations, combinations_with_replacement

from gym_splendor_code.envs.mechanics.game_settings import *
from gym_splendor_code.envs.mechanics.action import Action, ActionTradeGems, ActionBuyCard, ActionReserveCard
from gym_splendor_code.envs.mechanics.enums import GemColor
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.utils.utils_functions import tuple_of_gems_to_gems_collection


def generate_all_legal_actions(state: State) -> List[Action]:
    """Generates list of all possible actions of an active player in a given current_state."""
    return generate_all_legal_trades(state) + generate_all_legal_buys(state) + generate_all_legal_reservations(state)

def generate_all_legal_trades(state: State) -> List[ActionTradeGems]:
    """Returns the list of all possible actions of trade in a given current_state"""
    gems_board = state.board.gems_on_board
    gems_player = state.active_players_hand().gems_possessed

    list_of_actions_trade = []
    gems_player_n = sum(gems_player.jsonize())

    if gems_player_n < 8:
        """ Take 2 gems of same color """
        for color in gems_board.get_colors_on_condition(4):
            list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection({color},
                                                                                           val = 2)))
        """ Take 3 gems of different colors """
        for option in list(combinations(gems_board.get_colors_on_condition(1), 3)):
            list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option)))

    elif gems_player_n == 8:
        """ Take 2 gems of same color. """
        for color in gems_board.get_colors_on_condition(4):
            list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection({color}, 2)))
        """ Take 2 gems of different color. """
        for option in list(combinations(gems_board.get_colors_on_condition(1), 2)):
            list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option)))
        """ Take 3 gems of different color and return one of other color. """
        for option in list(combinations(gems_board.get_colors_on_condition(1), 3)):
            for option_return in (gems_player.get_all_colors_on_condition(1)).difference(option):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                          return_colors = {option_return})))
    elif gems_player_n == 9:
        """ Take 1 gem """
        for color in gems_board.get_colors_on_condition(1):
            list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection({color}
                                                                                                )))

        """ Take 2 gems of same color and return one of other color. """
        for color in gems_board.get_colors_on_condition(4):
            for option_return in (gems_player.get_all_colors_on_condition(1)).difference({color}):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection({color}, val = 2,
                                                                                          return_colors = {option_return})))
        """ Take 2 gems of different colors and return one of other color: """
        for option in list(combinations(gems_board.get_colors_on_condition(1), 2)):
            for option_return in (gems_player.get_all_colors_on_condition(1)).difference(option):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                          return_colors = {option_return})))
        """ Take 3 gems of different color and return: """
        for option in list(combinations(gems_board.get_colors_on_condition(1), 3)):
            """ 2 gems of different colors; """
            for option_return in list(combinations((gems_player.get_all_colors_on_condition(1)).difference(option), 2)):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                            return_val = [1,1],
                                                                                            return_colors = option_return)))
            """ 2 gems of same color. """
            for option_return in (gems_player.get_all_colors_on_condition(2)).difference(option):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                        return_val = [2],
                                                                                        return_colors = {option_return})))
    elif gems_player_n == 10:
        """ Exchange 1 gem """
        for color in gems_board.get_colors_on_condition(1):
            for option_return in (gems_player.get_all_colors_on_condition(1)).difference({color}):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection({color},
                                                                                               return_colors = {option_return})))

        """ Exchange 2 gems of same color: """
        for color in gems_board.get_colors_on_condition(4):
            """ for 2 gems of one but other color """
            for option_return in (gems_player.get_all_colors_on_condition(2)).difference({color}):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection({color},
                                                                                               val = 2, return_val = [2],
                                                                                               return_colors = {option_return})))
            """ for 2 gems of 2 other colors """
            for option_return in list(combinations((gems_player.get_all_colors_on_condition(1)).difference({color}), 2)):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection({color},
                                                                                               val = 2, return_val = [1,1],
                                                                                               return_colors = option_return)))

        """ Exchange 2 gems of two different colors: """
        for option in list(combinations(gems_board.get_colors_on_condition(1), 2)):
            """ for 2 gems of one but other color """
            for option_return in (gems_player.get_all_colors_on_condition(2)).difference(option):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                               return_val = [2],
                                                                                               return_colors = {option_return})))
            """ for 2 gems of 2 other colors """
            for option_return in list(combinations((gems_player.get_all_colors_on_condition(1)).difference(option), 2)):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                               return_val = [1,1],
                                                                                               return_colors = option_return)))
        """ Exchange 3 gems of different colors: """
        for option in list(combinations(gems_board.get_colors_on_condition(1), 3)):
            """ for 3 gems of 3 remaining colors """
            for option_return in list(combinations((gems_player.get_all_colors_on_condition(1)).difference(option), 3)):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                               return_val = [1,1,1],
                                                                                               return_colors = option_return)))
            """ for 2 gems of one and one of the remaining color """
            for option_return in (gems_player.get_all_colors_on_condition(2)).difference(option):
                for option_return_ in ((gems_player.get_all_colors_on_condition(1)).difference(option)).difference({option_return}):
                    list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                                   return_val = [2, 1],
                                                                                                   return_colors = [option_return, option_return_])))
            """ for 3 gems of one color """
            for option_return in (gems_player.get_all_colors_on_condition(3)).difference(option):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                               return_val = [3],
                                                                                               return_colors = {option_return})))
    return list_of_actions_trade


def generate_all_legal_buys(state: State) -> List[ActionBuyCard]:
    """Returns the list of all possible actions of buys in a given current_state"""
    list_of_actions_buy = []
    discount = state.active_players_hand().discount()
    all_cards_can_afford = [card for card in state.board.cards_on_board if
                            state.active_players_hand().can_afford_card(card, discount)] + \
                           [reserved_card for reserved_card in state.active_players_hand().cards_reserved if
                            state.active_players_hand().can_afford_card(reserved_card, discount)]

    for card in all_cards_can_afford:
        card_price_after_discount = card.price % discount
        minimum_gold_needed = state.active_players_hand().min_gold_needed_to_buy_card(card)
        for n_gold_gems_to_use in range(minimum_gold_needed,
                                        state.active_players_hand().gems_possessed.gems_dict[GemColor.GOLD] + 1):
            # we choose combination of other gems:
            options_of_use_gold_as = combinations_with_replacement(
                card_price_after_discount.non_empty_stacks_except_gold(),
                n_gold_gems_to_use)
            for option_of_use_gold_as in options_of_use_gold_as:
                use_gold_as = tuple_of_gems_to_gems_collection(option_of_use_gold_as)
                # check if the option satisfies conditions:
                condition_1 = use_gold_as <= card_price_after_discount
                condition_2 = state.active_players_hand().gems_possessed >= card_price_after_discount - use_gold_as
                if condition_1 and condition_2:
                    list_of_actions_buy.append(ActionBuyCard(card, n_gold_gems_to_use, use_gold_as))

    return list_of_actions_buy


def generate_all_legal_reservations(state: State) -> List[ActionReserveCard]:
    list_of_actions_reserve = []
    # first check if active player has not exceeded the limit of reservations
    condition_1 = len(state.active_players_hand().cards_reserved) < MAX_RESERVED_CARDS
    if condition_1:
        for card in state.board.cards_on_board:
            condition_2 = state.active_players_hand().gems_possessed.sum() < MAX_GEMS_ON_HAND
            condition_3 = state.board.gems_on_board.value(GemColor.GOLD) > 0
            if condition_2 and condition_3:
                # reserve card and take one golden gem for it:
                list_of_actions_reserve.append(ActionReserveCard(card, True))
            if not condition_3:
                # the are no golden gems on board, so reserve without taking golden gem:
                list_of_actions_reserve.append(ActionReserveCard(card, False))
            if condition_3 and not condition_2:
                # there are golden gems on board, but the player has reached the limit of gems on hand so can take one,
                # but must return one other:
                # 1. First case: do not take golden gem:
                list_of_actions_reserve.append(ActionReserveCard(card, False))
                # 2. Second case: take golden gem and return one other gem:
                for gem_color in state.active_players_hand().gems_possessed.non_empty_stacks_except_gold():
                    list_of_actions_reserve.append(ActionReserveCard(card, True, gem_color))

    return list_of_actions_reserve
