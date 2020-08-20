import random

print('-' * 50)
print('Welcome to Roulette')
print('-' * 50)
initial_balance_is_valid = False
while not initial_balance_is_valid:
    try:
        initial_balance = float(input('Let me see ,How much you have got   '))
        if initial_balance > 0:
            initial_balance_is_valid = True
        else:
            print("You can't bet 0 or negative amount of money !")
    except ValueError:
        print('Enter a valid number.')
print('-' * 50)
print("Great. Let's start.")
print('-' * 50)
game_is_playing = True
gains = []
while game_is_playing:
    bet_is_valid = False
    while bet_is_valid is not True:
        try:
            bet = float(input('How much money do you want to bet on this single roulette turn?   '))
            if bet <= initial_balance + sum(gains) and bet > 0:
                bet_is_valid = True
            else:
                print("You can't bet 0, negative amount of money or money that you don't have !")
        except ValueError:
            print('Please enter a valid number.')
    guess_number_is_valid = False
    while not guess_number_is_valid :
        try:
            guess_number = float(input('Guess the winning number.....   '))
            if 0 <= guess_number <= 49:
                guess_number_is_valid = True
            else:
                print("You have to enter a number between 0 and 49 !")
        except ValueError:
            print('Please enter a valid number.')
    winning_number = random.randint(0, 49)
    if random.randint(0, 1) == 0:
        black_or_red = 'black'
    else:
        black_or_red = 'red'
    if guess_number == winning_number:
        gain = bet * 3
    elif black_or_red == 'black' and 0 <= guess_number <= 24:
        gain = bet * 0.5
    else:
        gain = -bet
    if gain > 0:
        print('You won', gain)
        gains.append(gain)
    else:
        print('You loss', -gain)
        gains.append(gain)
    balance_final = initial_balance + sum(gains)
    print('Your balance is now', balance_final)
    if balance_final == 0:
        print('-' * 50)
        print('Sorry. You are bankrupt.')
        break
    replay_is_valid = False
    while not replay_is_valid :
        replay = input('Dare to bet again ?  yes or no')
        if replay == 'no':
            game_is_playing = False
            print('-' * 50)
            print('Initial balance:', initial_balance)
            if gain <=0:
                print('Loss: ', -sum(gains))
            else:
                print('Gain: ', sum(gains))
            print('Balance:', balance_final)
            print('-' * 50)
            print('See you next time.')
            break
        elif replay == 'yes' or replay == 'no':
            replay_is_valid = True
        elif replay == 'yes':
            game_is_playing = True

        else:
            print('Just enter yes or no.')