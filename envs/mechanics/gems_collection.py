from typing import Dict
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