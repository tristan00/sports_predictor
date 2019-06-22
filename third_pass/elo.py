
minimum_elo = 100
starting_elo = 1000
maximum_elo = 100000


def calculate_different_elos(outcome, player_1_elos, player_2_elos, k_values):
    results = {}
    for k in k_values:
        results[k] = calculate_new_elo(outcome, player_1_elos[k], player_2_elos[k], k)
    return results


def calculate_new_elo(outcome, player_1_elo, player_2_elo, k = 100):

    expected_outcome = player_1_elo/(player_1_elo + player_2_elo)
    opposite_of_outcome = 0 if outcome ==1 else 1
    new_elo = starting_elo
    if k > 0:
        new_elo =  player_1_elo + (k * (outcome-expected_outcome))

    #not elo, experiment to put streakiness
    elif k == 0:
        new_elo = player_1_elo * (1 + outcome - expected_outcome)
    elif k == -1:
        if outcome == 1:
            new_elo = min(player_1_elo * (1 + outcome - expected_outcome),
                          player_2_elo * (1 + outcome - (1 - expected_outcome)))
        else:
            new_elo = max(player_1_elo * (1 + outcome - expected_outcome),
                          player_2_elo * (1 + outcome - (1 - expected_outcome)))

    return min(maximum_elo, max(minimum_elo, new_elo))


def calculate_new_elo2(outcome, player_1_elo, player_2_elo):
    if player_1_elo != starting_elo and player_2_elo != starting_elo:
        pass
    expected_outcome = player_1_elo/(player_1_elo + player_2_elo)
    new_elo = player_1_elo * (1 + outcome - expected_outcome)

    return max(minimum_elo, new_elo)