import numpy as np

ACHIEVEMENTS=['collect_coal','collect_diamond','collect_drink','collect_iron','collect_sapling',
              'collect_stone','collect_wood','defeat_skeleton','defeat_zombie','eat_cow','eat_plant',
              'make_iron_pickaxe','make_iron_sword','make_stone_pickaxe','make_stone_sword',
              'make_wood_pickaxe','make_wood_sword','place_furnace','place_plant','place_stone',
              'place_table','wake_up']

def crafter_score(unlock_log):
    rates={n:100*np.mean([n in u for u in unlock_log]) for n in ACHIEVEMENTS}
    score=np.exp(np.mean(np.log(1+np.array(list(rates.values())))))-1
    return score,rates
