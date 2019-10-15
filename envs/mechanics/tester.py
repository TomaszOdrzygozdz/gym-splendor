# w1 = GemsCollecion({GemColor.GOLD: 1, GemColor.RED: 2, GemColor.GREEN: 5, GemColor.BLUE: 3, GemColor.WHITE: 4, GemColor.BLACK: 5})
# w2 = GemsCollecion({GemColor.GOLD: 2, GemColor.RED: 1, GemColor.GREEN: 6, GemColor.BLUE: 3, GemColor.WHITE: 7, GemColor.BLACK: 5})
# print(w1 <= w2)
from envs.graphics.splendor_gui import SplendorGUI
from envs.mechanics.state import State

s = State()
f = SplendorGUI()

card = s.board.cards_on_board.pop()
f.draw_card(card, 10, 20)
f.keep_window_open()