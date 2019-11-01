from typing import List
from itertools import combinations, combinations_with_replacement

from gym_splendor_code.envs.mechanics.game_settings import *
from gym_splendor_code.envs.mechanics.action import Action, ActionTradeGems, ActionBuyCard, ActionReserveCard
from gym_splendor_code.envs.mechanics.enums import GemColor
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.utils.utils_functions import tuple_of_gems_to_gems_collection


def generate_all_legal_actions(state: State) -> List[Action]:
    """Generates list of all possible actions of an active player in a given state."""
    return generate_all_legal_trades(state) + generate_all_legal_buys(state) + generate_all_legal_reservations(state)


def generate_all_legal_trades(state: State) -> List[ActionTradeGems]:
    """Returns the list of all possible actions of trade in a given state"""
    list_of_actions_trade = []
    n_non_empty_stacks = len(state.board.gems_on_board.non_empty_stacks_except_gold())
    n_gems_to_get_netto = min(MAX_GEMS_ON_HAND - state.active_players_hand().gems_possessed.sum(),
                              MAX_GEMS_IN_ONE_MOVE,
                              n_non_empty_stacks)

    max_gems_to_take = min(MAX_GEMS_IN_ONE_MOVE, n_non_empty_stacks)

    for n_gems_to_get in range(n_gems_to_get_netto, max_gems_to_take + 1):
        n_gems_to_return = n_gems_to_get - n_gems_to_get_netto

        # choose gems to get:
        options_of_taking = list(combinations(state.board.gems_on_board.non_empty_stacks_except_gold(), n_gems_to_get))
        for option_of_taking in options_of_taking:
            # now we have chosen which gems to take, so we need to decide which to return
            gem_colors_not_taken = {gem_color for gem_color in GemColor if gem_color not in option_of_taking}
            # find gems collection to take:
            gems_to_take = tuple_of_gems_to_gems_collection(option_of_taking)
            # find possible options of returning gems:
            options_of_returning = list(combinations_with_replacement(gem_colors_not_taken, n_gems_to_return))
            for option_of_returning in options_of_returning:
                # now we create gem collection describing transfer and check if it satisfies conditions of legal trade
                gems_to_return = tuple_of_gems_to_gems_collection(option_of_returning)

                gems_collection_to_trade = gems_to_take - gems_to_return
                # check if there is enough gems on the board to take:
                condition_1 = state.board.gems_on_board >= gems_collection_to_trade
                # check if the player has enough gems to return:
                condition_2 = state.active_players_hand().gems_possessed >= -gems_collection_to_trade
                if condition_1 and condition_2:
                    list_of_actions_trade.append(ActionTradeGems(gems_collection_to_trade))

    return list_of_actions_trade


def generate_all_legal_buys(state: State) -> List[ActionBuyCard]:
    """Returns the list of all possible actions of buys in a given state"""
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
