from typing import List
from itertools import combinations, combinations_with_replacement

from envs.data.game_settings import *
from envs.mechanics.action import Action, ActionTradeGems
from envs.mechanics.enums import GemColor
from envs.mechanics.gems_collection import GemsCollection
from envs.mechanics.players_hand import PlayersHand
from envs.mechanics.state import State


def give_all_legal_moves(state: State) -> List[Action]:
    """Returns the list of all possible actions in a given state"""

    list_of_actions_trade = []

    #We find moves of type trade gems:
    #_________________________________________________________________________________________________________________
    n_non_empty_stacks = len(state.board.gems_on_board.non_empty_stacks_except_gold())
    n_gems_to_get_netto =  min(MAX_GEMS_ON_HAND - state.active_players_hand().gems_possessed.sum(),
                               MAX_GEMS_IN_ONE_MOVE,
                               n_non_empty_stacks)

    max_gems_to_take = min(MAX_GEMS_IN_ONE_MOVE, n_non_empty_stacks)

    for n_gems_to_get in range(n_gems_to_get_netto, max_gems_to_take+1):
        n_gems_to_return = n_gems_to_get - n_gems_to_get_netto

        #choose gems to get:
        options_of_taking = list(combinations(state.board.gems_on_board.non_empty_stacks_except_gold(), n_gems_to_get))
        for option_of_taking in options_of_taking:
            #now we have chosen which gems to take, so we need to decide which to return
            gem_colors_not_taken = {gem_color for gem_color in GemColor if gem_color not in option_of_taking}
            # find gems collection to take
            gems_to_take_dict = {gem_color: 0 for gem_color in GemColor}
            for gem_ind in option_of_taking:
                gems_to_take_dict[gem_ind] += 1
            #find possible options of returning gems:
            options_of_returning = list(combinations_with_replacement(gem_colors_not_taken,n_gems_to_return))
            for option_of_returning in options_of_returning:
                #now we create gem collection describing transfer and check if it satisfies conditions of legal trade
                gems_to_return_dict = {gem_color: 0 for gem_color in GemColor}
                for gem_ind in option_of_returning:
                    gems_to_return_dict[gem_ind] += 1

                gems_collection_to_trade = GemsCollection(gems_to_take_dict) - GemsCollection(gems_to_return_dict)

                #check if there is enough gems on the board to take:
                condition_1 = state.board.gems_on_board >= gems_collection_to_trade
                #check if the player has enough gems to return:
                condition_2 = state.active_players_hand().gems_possessed >= -gems_collection_to_trade

                if condition_1 and condition_2:
                    list_of_actions_trade.append(ActionTradeGems(gems_collection_to_trade))

    return list_of_actions_trade


#testing
pla = PlayersHand()
pla.gems_possessed.gems_dict[GemColor.BLUE] = 9
d = State()
d.list_of_players_hands = [pla, PlayersHand()]
f = give_all_legal_moves(d)
print(len(f))
for du  in f:
    print(du)
