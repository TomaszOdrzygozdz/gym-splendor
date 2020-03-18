from typing import List
from itertools import combinations, combinations_with_replacement

from gym_splendor_code.envs.mechanics.game_settings import *
from gym_splendor_code.envs.mechanics.action import Action, ActionTradeGems, ActionBuyCard, ActionReserveCard
from gym_splendor_code.envs.mechanics.enums import GemColor
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.utils.utils_functions import tuple_of_gems_to_gems_collection, colors_to_gems_collection


def generate_all_legal_trades_fast(state: State) -> List[ActionTradeGems]:
    """Returns the list of all possible actions of trade in a given current_state"""
    gems_board = state.board.gems_on_board
    gems_player = state.active_players_hand().gems_possessed
    #print(state.active_players_hand().discount)
    n_non_empty_stacks = len(gems_board.non_empty_stacks_except_gold())
    take3 = min(3, n_non_empty_stacks)
    take2 = min(2, n_non_empty_stacks)

    list_of_actions_trade = []
    gems_player_n = sum(gems_player.to_dict())

    if gems_player_n < 8:
        """ Take 2 gems of same color """
        for color in gems_board.get_colors_on_condition(4):
            list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection({color},
                                                                                           val = 2)))
        """ Take 3 gems of different colors """
        for option in list(combinations(gems_board.get_colors_on_condition(1), take3)):
            list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option)))

    elif gems_player_n == 8:
        """ Take 2 gems of same color. """
        for color in gems_board.get_colors_on_condition(4):
            list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection({color}, 2)))
        """ Take 2 gems of different color. """
        for option in list(combinations(gems_board.get_colors_on_condition(1), take2)):
            list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option)))
        """ Take 3 gems of different color and return one of other color. """
        for option in list(combinations(gems_board.get_colors_on_condition(1), take3)):
            for option_return in (gems_player.get_all_colors_on_condition(1)).difference(list(option)):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                          return_colors = {option_return})))
    elif gems_player_n == 9:
        """ Take 1 gem """
        for color in gems_board.get_colors_on_condition(1):
            list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection({color})))

        """ Take 2 gems of same color and return one of other color. """
        for color in gems_board.get_colors_on_condition(4):
            for option_return in (gems_player.get_all_colors_on_condition(1)).difference(list({color})):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection({color}, val = 2,
                                                                                          return_colors = {option_return})))
        """ Take 2 gems of different colors and return one of other color: """
        for option in list(combinations(gems_board.get_colors_on_condition(1), 2)):
            for option_return in (gems_player.get_all_colors_on_condition(1)).difference(list(option)):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                          return_colors = {option_return})))
        """ Take 3 gems of different color and return: """
        for option in list(combinations(gems_board.get_colors_on_condition(1), 3)):
            """ 2 gems of different colors; """
            for option_return in list(combinations((gems_player.get_all_colors_on_condition(1)).difference(list(option)), 2)):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                            return_val = [1,1],
                                                                                            return_colors = option_return)))
            """ 2 gems of same color. """
            for option_return in (gems_player.get_all_colors_on_condition(2)).difference(list(option)):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                        return_val = [2],
                                                                                        return_colors = {option_return})))
    elif gems_player_n == 10:
        """ Exchange 1 gem """
        for color in gems_board.get_colors_on_condition(1):
            for option_return in (gems_player.get_all_colors_on_condition(1)).difference(list({color})):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection({color},
                                                                                               return_colors = {option_return})))

        """ Exchange 2 gems of same color: """
        for color in gems_board.get_colors_on_condition(4):
            """ for 2 gems of one but other color """
            for option_return in (gems_player.get_all_colors_on_condition(2)).difference(list({color})):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection({color},
                                                                                               val = 2, return_val = [2],
                                                                                               return_colors = {option_return})))

            """ for 2 gems of 2 other colors """
            for option_return in list(combinations((gems_player.get_all_colors_on_condition(1)).difference(list({color})), 2)):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection({color},
                                                                                               val = 2, return_val = [1,1],
                                                                                               return_colors = option_return)))

        """ Exchange 2 gems of two different colors: """
        for option in list(combinations(gems_board.get_colors_on_condition(1), 2)):
            """ for 2 gems of one but other color """
            for option_return in (gems_player.get_all_colors_on_condition(2)).difference(list(option)):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                               return_val = [2],
                                                                                               return_colors = {option_return})))

            """ for 2 gems of 2 other colors """
            for option_return in list(combinations((gems_player.get_all_colors_on_condition(1)).difference(list(option)), 2)):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                               return_val = [1,1],
                                                                                               return_colors = option_return)))

        """ Exchange 3 gems of different colors: """
        for option in list(combinations(gems_board.get_colors_on_condition(1), 3)):
            """ for 3 gems of 3 remaining colors """
            for option_return in list(combinations((gems_player.get_all_colors_on_condition(1)).difference(list(option)), 3)):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                               return_val = [1,1,1],
                                                                                               return_colors = option_return)))

            """ for 2 gems of one and one of the remaining color """
            for option_return in (gems_player.get_all_colors_on_condition(2)).difference(list(option)):
                for option_return_ in ((gems_player.get_all_colors_on_condition(1)).difference(list(option))).difference(list({option_return})):
                    list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                                   return_val = [2, 1],
                                                                                                   return_colors = [option_return, option_return_])))

            """ for 3 gems of one color """
            for option_return in (gems_player.get_all_colors_on_condition(3)).difference(list(option)):
                list_of_actions_trade.append(ActionTradeGems(tuple_of_gems_to_gems_collection(option,
                                                                                               return_val = [3],
                                                                                               return_colors = {option_return})))

    return list_of_actions_trade


def generate_all_legal_buys_fast(state: State) -> List[ActionBuyCard]:
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

def generate_all_legal_buys_fast_need_testing(state: State) -> List[ActionBuyCard]:
    """Returns the list of all possible actions of buys in a given current_state"""
    list_of_actions_buy = []
    discount = state.active_players_hand().discount()
    all_cards_can_afford = [card for card in state.board.cards_on_board if
                            state.active_players_hand().can_afford_card(card, discount)] + \
                           [reserved_card for reserved_card in state.active_players_hand().cards_reserved if
                            state.active_players_hand().can_afford_card(reserved_card, discount)]

    for card in all_cards_can_afford:
        new_price = card.price % state.active_players_hand().discount()
        gems = state.active_players_hand().gems_possessed
        minimum_gold_needed = state.active_players_hand().min_gold_needed_to_buy_card(card)

        for n_gold_gems_to_use in range(minimum_gold_needed,
                                        state.active_players_hand().gems_possessed.gems_dict[GemColor.GOLD]):
            if n_gold_gems_to_use == 0:
                list_of_actions_buy.append(ActionBuyCard(card))
            else:
                if n_gold_gems_to_use == 1:
                    colors_1 = new_price.get_colors_on_condition(1).intersection(gems.get_colors_on_condition(1))
                    for color1 in colors_1:
                        list_of_actions_buy.append(ActionBuyCard(card, 1, colors_to_gems_collection([color1])))


                elif n_gold_gems_to_use == 2:
                    colors_1 = new_price.get_colors_on_condition(1).intersection(gems.get_colors_on_condition(1))
                    colors_2 = new_price.get_colors_on_condition(2).intersection(gems.get_colors_on_condition(2))

                    for option in colors_2:
                        list_of_actions_buy.append(ActionBuyCard(card, 2, colors_to_gems_collection([option, option])))
                    for option in combinations(colors_1, 2):
                        list_of_actions_buy.append(ActionBuyCard(card, 2, colors_to_gems_collection(list(option))))

                elif n_gold_gems_to_use == 3:
                    colors_1 = new_price.get_colors_on_condition(1).intersection(gems.get_colors_on_condition(1))
                    colors_2 = new_price.get_colors_on_condition(2).intersection(gems.get_colors_on_condition(2))
                    colors_3 = new_price.get_colors_on_condition(3).intersection(gems.get_colors_on_condition(3))

                    for option in colors_3:
                        list_of_actions_buy.append(ActionBuyCard(card, 3,
                        colors_to_gems_collection([option, option, option])))
                    for option in colors_2:
                        for option_2 in colors_1.difference({option}):
                            list_of_actions_buy.append(ActionBuyCard(card, 3,
                            colors_to_gems_collection([option, option, option_2])))
                    for option in combinations(colors_1, 3):
                        list_of_actions_buy.append(ActionBuyCard(card, 3,
                         colors_to_gems_collection(list(option))))


                elif n_gold_gems_to_use == 4:
                    colors_1 = new_price.get_colors_on_condition(1).intersection(gems.get_colors_on_condition(1))
                    colors_2 = new_price.get_colors_on_condition(2).intersection(gems.get_colors_on_condition(2))
                    colors_3 = new_price.get_colors_on_condition(3).intersection(gems.get_colors_on_condition(3))
                    colors_4 = new_price.get_colors_on_condition(4).intersection(gems.get_colors_on_condition(4))

                    for option in colors_4:
                        list_of_actions_buy.append(ActionBuyCard(card, 4,
                        colors_to_gems_collection([option, option, option, option])))
                    for option in colors_3:
                        for option_2 in colors_1.difference(list(option)):
                            list_of_actions_buy.append(ActionBuyCard(card, 4,
                            colors_to_gems_collection([option, option, option, option_2])))
                    for option in colors_2:
                        for option_2 in combinations(colors_1.difference(list(option)), 2):
                            list_of_actions_buy.append(ActionBuyCard(card, 4,
                            colors_to_gems_collection([option, option].extend(list(option_2)))))
                    for option in combinations(colors_2, 2):
                        list_of_actions_buy.append(ActionBuyCard(card, 4,
                        colors_to_gems_collection(list(option).extend(list(option_2)))))
                    for option in combinations(colors_1, 4):
                        list_of_actions_buy.append(ActionBuyCard(card, 4,
                        colors_to_gems_collection(list(option))))


                elif n_gold_gems_to_use == 5:
                    colors_1 = new_price.get_colors_on_condition(1).intersection(gems.get_colors_on_condition(1))
                    colors_2 = new_price.get_colors_on_condition(2).intersection(gems.get_colors_on_condition(2))
                    colors_3 = new_price.get_colors_on_condition(3).intersection(gems.get_colors_on_condition(3))
                    colors_4 = new_price.get_colors_on_condition(4).intersection(gems.get_colors_on_condition(4))
                    colors_5 = new_price.get_colors_on_condition(5).intersection(gems.get_colors_on_condition(5))

                    for option in colors_5:
                        list_of_actions_buy.append(ActionBuyCard(card, 5,
                        colors_to_gems_collection([option, option, option, option, option])))
                    for option in colors_4:
                        for option_2 in colors_1.difference(list(option)):
                            list_of_actions_buy.append(ActionBuyCard(card, 5,
                            colors_to_gems_collection([option, option, option, option, option_2])))
                    for option in colors_3:
                        for option_2 in colors_2.difference(list(option)):
                            list_of_actions_buy.append(ActionBuyCard(card, 5,
                            colors_to_gems_collection([option, option, option, option_2, option_2])))
                        for option_2 in combinations(colors_1.difference(list(option)), 2):
                            list_of_actions_buy.append(ActionBuyCard(card, 5,
                            colors_to_gems_collection([option, option, option].extend(list(option_2)))))

                    for option in combinations(colors_2, 2):
                        for option_2 in colors_1.difference(set(option)):
                            list_of_actions_buy.append(ActionBuyCard(card, 5,
                            colors_to_gems_collection([option_2].extend(2 * list(option)))))
                    for option in colors_2:
                        for option_2 in combinations(colors_1.difference({option}), 3):
                            list_of_actions_buy.append(ActionBuyCard(card, 5,
                            colors_to_gems_collection([option, option].extend(list(option_2)))))

                    for option in combinations(colors_1, 5):
                        list_of_actions_buy.append(ActionBuyCard(card, 5,
                        colors_to_gems_collection(list(option))))

    return list_of_actions_buy


if ALLOW_RESERVATIONS:

    def generate_all_legal_reservations_fast(state: State) -> List[ActionReserveCard]:
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

if not ALLOW_RESERVATIONS:
    def generate_all_legal_reservations_fast(state: State) -> List[ActionReserveCard]:
        return []