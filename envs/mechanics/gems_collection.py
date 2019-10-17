from typing import Dict, Set
from functools import reduce
from envs.mechanics.enums import GemColor

class GemsCollection():
    '''This class is used to desribed a collection of gemes. It can be treated both as a wallet of games or as a price
     of a single card.'''

    def __init__(self,
                 gems_values_dict: Dict[GemColor, int] = None) -> None:
        """Creates a collection of gems.
        Parameters:
        _ _ _ _ _ _
        gems_values: Dictionary with the keys being gem color ans values being integers. Ig gems=None, than it creates a
        collection with all values set to zero."""

        if gems_values_dict is None:
            self.gems_dict = {gem_color : 0 for gem_color in GemColor}
        else:
            self.gems_dict = gems_values_dict

    def value(self, gem_color):
        """Returns value of gem_color form this gems collection."""
        return self.gems_dict[gem_color]

    def sum(self):
        return sum(self.gems_dict.values())

    def __add__(self, other):
        """Adds other gems colletion to this one.

        Parameters:
        _ _ _ _ _ _
        other: An object of class GemsCollection to be added.

        Returns:
        _ _ _ _ _ _
        An object of class GemsCollection with a dictionary of color values, which is a key-wise sum.
        """
        return GemsCollection({gem_color : self.gems_dict[gem_color] + other.gems_dict[gem_color] for gem_color in GemColor})

    def __sub__(self, other):
        """Subtracts other gems colletion from this one.

        Parameters:
        _ _ _ _ _ _
        other: An object of class GemsCollection to be subtracted.

        Returns:
        _ _ _ _ _ _
        An object of class GemsCollection with a dictionary of color values, which is a key-wise sum.
        """
        return GemsCollection({gem_color : self.gems_dict[gem_color] - other.gems_dict[gem_color] for gem_color in GemColor})

    def __le__(self, other):
        """Checks if this instance is smaller or equal to the other (gem-wise check).

        Parameters:
        _ _ _ _ _ _
        other: An object of class GemsCollection to be compared.

        Returns:
        _ _ _ _ _ _
        Boolean value, that is True if for each gem color the value of this on this color is <= than the value of other
        on this color.
        """
        return reduce(lambda x, y: x and y, [self.gems_dict[gem_color] <= other.gems_dict[gem_color] for gem_color in
                                             GemColor])

    def __ge__(self, other):
        """Checks if this instance is greater or equal to the other (gem-wise check).

        Parameters:
        _ _ _ _ _ _
        other: An object of class GemsCollection to be compared.

        Returns:
        _ _ _ _ _ _
        Boolean value, that is True if for each gem color the value of this on this color is >= than the value of other
        on this color.
        """
        return reduce(lambda x, y: x and y, [self.gems_dict[gem_color] >= other.gems_dict[gem_color] for gem_color in
                                             GemColor])

    def __mod__(self, other):
        """Subtracts other gems collection form self and then sets all negative values to zero.

        Parameters:
        _ _ _ _ _ _
        other: An object of class GemsCollection to be subtracted and later made non-negative.

        Returns:
        _ _ _ _ _ _
        Gems collection that is the result of this operation."""

        return GemsCollection({gem_color : max(0, self.gems_dict[gem_color] - other.gems_dict[gem_color]) for
                               gem_color in GemColor})

    def __iadd__(self, other):
        self = self.__add__(other)

    def __isub__(self, other):
        self = self.__sub__(other)

    def __neg__(self):
        """Returns gems collection with -values for each gem color."""
        return GemsCollection({gem_color: -self.gems_dict[gem_color] for gem_color in GemColor})

    def __repr__(self):
        return self.gems_dict.__repr__().replace('GemColor.','')

    def non_empty_stacks(self) -> Set[GemColor]:
        return {gem_color for gem_color in GemColor if self.gems_dict[gem_color] > 0}

    def non_empty_stacks_except_gold(self) -> Set[GemColor]:
        return {gem_color for gem_color in GemColor if self.gems_dict[gem_color] > 0
                and gem_color != GemColor.GOLD}

