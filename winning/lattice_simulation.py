from winning.lattice_conventions import STD_UNIT, STD_SCALE, STD_L, STD_A, NAN_DIVIDEND
from winning.lattice_calibration import dividend_implied_ability, normalize_dividends, dividends_from_prices
from winning.lattice import skew_normal_density, densities_from_offsets, pdf_to_cdf, sample_from_cdf
import heapq
from collections import Counter
import numpy as np

# A few utilities for simulating from a race where runner performance distributions are represented on a lattice.
# None of this is essential to the core calibration algorithm.
# Rather, it is used post-calibration to Monte Carlo prices for win, place, show and exotics.
# No claims of computational efficiency here!


PLACING_NAMES = ['win','place2','place3','place4','place5','place6',
                 'place7','place8','place9','place10','place11',
                 'place12','place13','place14']   # aka win/place/show/top4
N_SAMPLES = 5000                                     # Default number of Monte Carlo paths


def simulate_performances(densities, n_samples:int=N_SAMPLES, unit=1.0, add_noise=True):
    """ Simulate multiple contest outcomes
    :param densities:
    :param n_samples:
    :return:  [ [race performance] ]    List of races
    """
    cdfs = [pdf_to_cdf(density) for density in densities]
    cols = [sample_from_cdf(cdf, n_samples=n_samples,add_noise=add_noise,unit=unit) for cdf in cdfs]
    rows = list(map(list, zip(*cols)))
    return rows


def placegetters_from_performances(performances, n=4) -> [[int]]:
    """
    :param performances:  List of list of performances
    :return:  List of Lists
    """
    return [[placegetter(row, k) for row in performances] for k in range(n) ]


def placegetter(scores:[float], position:int):
    """ Return the index of the participant finishing in position+1
    :param scores:
    :param position:  0 for first place, 1 for second etc
    :return:
    """
    return heapq.nsmallest(position + 1, range(len(scores)), key=scores.__getitem__)[position]


def skew_normal_place_pricing(dividends, n_samples=N_SAMPLES, longshot_expon:float=1.0, a:float=STD_A, scale=STD_SCALE, nan_value=NAN_DIVIDEND, loc=0) -> dict:
    """ Price place/show and exotics from win market by Monte Carlo of performances
        :param  dividends  [ float ] decimal prices
        :param  longshot_expon  power law to apply to dividends, if you want to try to correct for longshot bias.
        :param  a         skew parameter in skew-normal running time distribution
        :param  scale     scale parameter in skew-normal running time distribution
        :returns  {'win':[1.6,4.5,...], 'place':[  ] , ... }
    """
    # TODO: Add control variates
    unit = STD_UNIT
    L = STD_L
    density = skew_normal_density(L=L, unit=unit, scale=scale, a=a, loc=loc)
    adj_dividends = longshot_adjusted_dividends(dividends=dividends,longshot_expon=longshot_expon)
    offsets = dividend_implied_ability(dividends=adj_dividends, density=density,nan_value=nan_value)
    densities = densities_from_offsets(density=density, offsets=offsets)
    performances = simulate_performances(densities=densities, n_samples=n_samples, add_noise=True, unit=unit)
    placegetters = placegetters_from_performances(performances=performances, n=14)
    the_counts = exotic_count(placegetters, do_exotics=False)
    n_runners = len(adj_dividends)
    prices = dict()
    for bet_type, multiplicity in zip(PLACING_NAMES,range(1,15)):
        prices[bet_type] = dividends_from_prices( [the_counts[bet_type][j] for j in range(n_runners)], multiplicity=multiplicity)
    return prices


def longshot_adjusted_dividends(dividends,longshot_expon=1.17):
    """ Use power law to approximately unwind longshot effect
        Obviously this is market dependent
    """
    dividends = [(o + 1.0) ** (longshot_expon) for o in dividends]
    return normalize_dividends(dividends)


def exotic_count(placegetters, do_exotics=False):
    """  Given counters for winner, second place etc, create counters for win,place,show and exotics """
    # A tad ugly :)
    winner, second, third, forth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, twelfth, thirtheenth, fourteenth = placegetters[0], placegetters[1],placegetters[2], placegetters[3], placegetters[4], placegetters[5], placegetters[6], placegetters[7], placegetters[8], placegetters[9], placegetters[10], placegetters[11], placegetters[12], placegetters[13]
    win = Counter(winner)
    place = Counter(second)
    place.update(win)
    show = Counter(third)
    show.update(place)
    top4 = Counter(forth)
    top4.update(show)
    
    top5 = Counter(fifth)
    top5.update(show)
    
    top6 = Counter(sixth)
    top6.update(show)
    
    top7 = Counter(seventh)
    top7.update(show)
    
    top8 = Counter(eighth)
    top8.update(show)
    
    top9 = Counter(ninth)
    top9.update(show)
    
    top10 = Counter(tenth)
    top10.update(show)
    
    top11 = Counter(eleventh)
    top11.update(show)
    
    top12 = Counter(twelfth)
    top12.update(show)
    
    top13 = Counter(thirtheenth)
    top13.update(show)
    
    top14 = Counter(fourteenth)
    top14.update(show)
    
    if do_exotics:
        exacta = Counter(zip(winner, second))
        trifecta = Counter(zip(winner, second, third))
        pick4 = Counter(zip(winner,second,third,forth))
        pick5 = Counter(zip(winner,second,third,forth,fifth))
        pick6 = Counter(zip(winner,second,third,forth,fifth,sixth))
        pick7 = Counter(zip(winner,second,third,forth,fifth,sixth,seventh))
        pick8 = Counter(zip(winner,second,third,forth,fifth,sixth,seventh,eighth))
        pick9 = Counter(zip(winner,second,third,forth,fifth,sixth,seventh,eighth,ninth))
        pick10 = Counter(zip(winner,second,third,forth,fifth,sixth,seventh,eighth,ninth,tenth))
        pick11 = Counter(zip(winner,second,third,forth,fifth,sixth,seventh,eighth,ninth,tenth,eleventh))
        pick12 = Counter(zip(winner,second,third,forth,fifth,sixth,seventh,eighth,ninth,tenth,eleventh,twelfth))
        pick13 = Counter(zip(winner,second,third,forth,fifth,sixth,seventh,eighth,ninth,tenth,eleventh,twelfth,thirtheenth))
        pick14 = Counter(zip(winner,second,third,forth,fifth,sixth,seventh,eighth,ninth,tenth,eleventh,twelfth,thirtheenth,fourteenth))
    else:
        exacta, trifecta, pick4, pick5, pick6, pick7, pick8, pick9, pick10, pick11, pick12, pick13, pick14 = None, None, None, None, None, None, None, None, None, None, None, None, None
    return {"win": win, "place2": place, "place3": show,"place4":top4,
            "place5":top5, "place6":top6, "place7":top7, "place8":top8,
            "place9":top9, "place10":top10, "place11":top11, "place12":top12,
            "place13":top13, "place14":top14,
            "exacta": exacta, "trifecta": trifecta, "pick4":pick4, "pick5":pick5,
            "pick6":pick6, "pick7":pick7, "pick8":pick8, "pick9":pick9, "pick10":pick10,
            "pick11":pick11, "pick12":pick12, "pick13":pick13, "pick14":pick14}


if __name__ == '__main__':
    # An illustration...
    derby = {'Essential Quality': 2,
             'Rock Your World': 5,
             'Known Agenda': 8,
             'Highly Motivated': 10,
             'Hot Rod Charlie': 10,
             'Medina Spirit': 16,
             'Mandaloun': 16,
             'Dynamic One': 20,
             'Bourbonic': 25,
             'Midnight Bourbon': np.nan,  # <--- This will be ignored
             'Super Stock': 25,
             'Soup and Sandwich': 33,
             'O Besos': 33,
             'King Fury': 33,
             'Helium': 33,
             'Like The King': 40,
             'Brooklyn Strong': 50,
             'Keepmeinmind': 50,
             'Hidden Stash': 50,
             'Sainthood': 50}
    if False:
        dividends = [ o+1.0 for o in derby.values() ]
    else:
        dividends = [6,6,6,6,6,6]
    prices = skew_normal_place_pricing(dividends=dividends, longshot_expon=1.17, n_samples=5000)
    from pprint import pprint
    pprint(list(zip(prices['win'],prices['place3'])))
